import os
import glob
import numpy as np
from tqdm import tqdm
import argparse

# --- Start of Wav2Vec2Phoneme - Audio to Phoneme code ---
# Ensure transformers, torchaudio, and librosa are installed:
# pip install transformers torchaudio librosa
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
try:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Wav2Vec2 model loaded and moved to {device}")
except Exception as e:
    print(f"Error loading Wav2Vec2 model or processor: {e}")
    print("Please ensure you have an active internet connection or the model is cached locally.")
    print("Also, check if 'transformers', 'torchaudio', and 'librosa' are installed.")
    exit() # Exit if model cannot be loaded, as it's critical

def audio_to_phone(audio_file_path):
    """
    Processes an audio file using Wav2Vec2 to extract phonemes and logits.
    Returns the decoded phonemes and the raw logits as a NumPy array.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)

        # Handle stereo audio by converting to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if necessary (Wav2Vec2's expected sample rate)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        # Move input tensor to the same device as the model
        inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode phonemes (optional, but good for verification)
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids, group_tokens=True)

        # Squeeze batch dimension, move to CPU, transpose for desired shape, convert to NumPy
        logits_squeezed = logits.squeeze(0).cpu() # Shape: (time_steps, vocab_size)
        # No need to transpose if you want (time_steps, vocab_size), which is typical for sequence models
        # If you specifically need (vocab_size, time_steps), then transpose: logit_trans = logits_squeezed.transpose(0,1)
        logit_trans = logits_squeezed.transpose(0,1)
        # logits_squeezed = logits_squeezed[0]
        logits_np_arr = logit_trans.numpy() # THIS IS THE CHANGE
        
        # print(type(logits_np_arr) , logits_np_arr.shape)

        return phonemes, logits_np_arr
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {e}")
        return None, None # Return None on error

# --- End of Wav2Vec2Phoneme code ---

def main():
    parser = argparse.ArgumentParser(description='Extract phoneme logits using Wav2Vec2 and organize data.')
    parser.add_argument('--audio_root_dir', type=str, required=True,
                        help="Path to the parent directory containing language folders (e.g., /path/to/my_data/train)")
    parser.add_argument('--project_root_dir', type=str, required=True,
                        help="Root directory where the new .npy feature folder and list files will be created.")

    args = parser.parse_args()

    # Determine the dataset type (e.g., 'train', 'test') from the parent directory name
    dataset_type = os.path.basename(os.path.normpath(args.audio_root_dir))

    # Define output paths dynamically
    phoneme_npy_feat_folder = os.path.join(args.project_root_dir, f"{dataset_type}_phoneme_npy_featrs")
    list_folder = os.path.join(args.project_root_dir, f"phoneme_npy_feat_list")
    final_list_path = os.path.join(list_folder, f"{dataset_type}_list2.txt")

    # Create output directories if they don't exist
    os.makedirs(phoneme_npy_feat_folder, exist_ok=True)
    os.makedirs(list_folder, exist_ok=True)

    # Discover language folders and create a consistent ID mapping
    language_folders = [d for d in os.listdir(args.audio_root_dir) if os.path.isdir(os.path.join(args.audio_root_dir, d))]
    # Sort for consistent ID assignment
    language_folders.sort()
    language_to_id = {lang.lower(): i for i, lang in enumerate(language_folders)}
    print("--- Language to ID Mapping ---")
    for lang, idx in language_to_id.items():
        print(f"{lang}: {idx}")
    print("------------------------------\n")

    # Open the final list file to write throughout the process
    with open(final_list_path, "w") as f_final_list:
        print(f"Starting phoneme logit extraction. Features will be saved to: {phoneme_npy_feat_folder}")
        print(f"The final list (wav2vec2_phoneme_all_list.txt) will be saved to: {final_list_path}\n")

        for lang_folder_name in tqdm(language_folders, desc="Processing Languages"):
            current_lang_id = language_to_id.get(lang_folder_name.lower())
            if current_lang_id is None:
                print(f"Warning: Language folder '{lang_folder_name}' not found in mapping. Skipping.")
                continue

            lang_path = os.path.join(args.audio_root_dir, lang_folder_name)
            audio_files = glob.glob(os.path.join(lang_path, "*.wav"))

            print(f"  -> Processing {len(audio_files)} audio files in '{lang_folder_name}'...")

            for audio_file_path in tqdm(audio_files, desc=f"    Extracting from {lang_folder_name}"):
                # Derive filename for .npy save
                filename_base = os.path.basename(audio_file_path).replace(".wav", "")
                npy_save_path = os.path.join(phoneme_npy_feat_folder, f"{filename_base}.npy")

                # Perform phoneme logit extraction
                phonemes, logits_np_arr = audio_to_phone(audio_file_path)
                

                if logits_np_arr is not None:
                    try:
                        np.save(npy_save_path, logits_np_arr)
                        f_final_list.write(f"{npy_save_path} {current_lang_id}\n")
                    except Exception as e:
                        print(f"Error saving {npy_save_path}: {e}")
                else:
                    print(f"Skipping {audio_file_path} due to extraction error.")

    print(f"\nFinished phoneme logit extraction and list generation.")
    print(f"All phoneme logits saved to: {phoneme_npy_feat_folder}")
    print(f"Final list generated at: {final_list_path}")

if __name__ == "__main__":
    main()