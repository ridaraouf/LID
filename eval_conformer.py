import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from model import Conformer
from data_load import RawFeatures, collate_fn_atten, get_atten_mask
from sklearn.metrics import accuracy_score
import compute_eer
import scoring

# ----- CONFIG -----
MODEL_PATH = "/home/teaching/Transformermd_phonemes11_22_04_4.ckpt"   # path to saved model
TEST_LIST = "/home/teaching/conformer_mfcc/phoneme_npy_feat_list/youtube_copy_list2.txt"                             # test file: each line -> feat.npy <label>
INPUT_DIM = 392
N_LANG = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------

def main():
    model = Conformer(
        input_dim=INPUT_DIM, feat_dim=32, d_k=32, d_v=32, n_heads=4,
        d_ff=1024, max_len=100000, dropout=0.1, device=DEVICE, n_lang=N_LANG
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    test_set = RawFeatures(TEST_LIST)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn_atten)

    all_scores = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for utt, labels, seq_len in tqdm(loader, desc="Evaluating"):
            utt = utt.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = get_atten_mask(seq_len, utt.size(0)).to(DEVICE)

            logits = model(utt, mask)
            probs = F.softmax(logits, dim=1)

            all_scores.append(probs.cpu())
            all_preds.append(torch.argmax(probs, dim=1).cpu())
            all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(all_labels, all_preds)
    print(f"Classification Accuracy: {acc * 100:.2f}%")

    # Score/EER Evaluation
    score_txt = "score_conformer.txt"
    trial_txt = "trial_conformer.txt"
    scoring.get_score(TEST_LIST, all_scores, N_LANG, score_txt)
    scoring.get_trials(TEST_LIST, N_LANG, trial_txt)

    target, non_target, _ = compute_eer.load_file(score_txt, trial_txt)
    p_miss, p_fa = compute_eer.compute_rocch(target, non_target)
    eer = compute_eer.rocch2eer(p_miss, p_fa)
    cavg = scoring.compute_cavg(trial_txt, score_txt)

    print(f"EER: {eer:.4f}")
    print(f"Cavg: {cavg:.4f}")

if __name__ == "__main__":
    main()
