import torch
import librosa
import numpy as np
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import glob
from tqdm import tqdm

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Wav2Vec2 feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
model.eval()


lang_labels = {
    'asm': '0', 'ben': '1', 'eng': '2', 'guj': '3',
    'hin': '4', 'kan': '5', 'mal': '6', 'mar': '7',
    'pun': '8', 'tam': '9', 'tel': '10'
}

# Feature extraction function
def extract_w2v2_features(wav_path, feature_dir, index_file, name, layer=12):
    label = lang_labels.get(name)
    if label is None:
        print(f"Skipping unknown or unsupported language: {name}")
        return

    os.makedirs(feature_dir, exist_ok=True)

    try:
        waveform, _ = librosa.load(wav_path, sr=16000)
        inputs = feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        features = outputs.hidden_states[layer].cpu().numpy().squeeze().T

        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        feature_path = os.path.join(feature_dir, f"{base_name}_w2v2.npy")
        np.save(feature_path, features)

        with open(index_file, "a") as f:
            f.write(f"{feature_path} {label}\n")

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")


# Main script
def main():
    # ✅ Input and output paths
    input_root = "/home/teaching/conformer_mfcc/split_dataset_copy/train_pre"
    output_root = "/home/teaching/conformer_mfcc/phoneme_npy_feat_list"
    feature_root = os.path.join(output_root, "features_train")
    index_file = os.path.join(output_root, "index_train.txt")
    layer = 12

    os.makedirs(feature_root, exist_ok=True)

    # Clear/create index.txt
    with open(index_file, "w") as f:
        f.write("")

    # Iterate over each language folder
    for lang_folder in os.listdir(input_root):
        if lang_folder not in lang_labels:
            print(f"Skipping unrecognized language folder: {lang_folder}")
            continue

        lang_path = os.path.join(input_root, lang_folder)
        if not os.path.isdir(lang_path):
            continue

        print(f"\n🔍 Processing language: {lang_folder}")

        lang_feature_dir = os.path.join(feature_root, lang_folder)
        os.makedirs(lang_feature_dir, exist_ok=True)

        wav_files = glob.glob(os.path.join(lang_path, "*.wav"))

        for wav_file in tqdm(wav_files, desc=f"{lang_folder}"):
            extract_w2v2_features(
                wav_path=wav_file,
                feature_dir=lang_feature_dir,
                index_file=index_file,
                name=lang_folder,
                layer=layer
            )


if __name__ == "__main__":
    main()
