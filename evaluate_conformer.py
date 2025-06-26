import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# your model definitions
#from model import X_Transformer_E2E_LID, Transformer_E2E_LID, Conformer
from model import Transformer_E2E_LID
#############################
# EER computation (same as before)
#############################
def compute_eer(pairs_2, numP, numN):
    all_scores = sorted(pairs_2, key=lambda x: x[0])
    numFA, numFR = numN, 0
    best_eer, best_thresh = 0.0, 0.0
    memory = []
    for score, is_target in all_scores:
        if is_target:
            numFR += 1
        else:
            numFA -= 1
        far = numFA / numN
        frr = numFR / numP
        if far <= frr:
            delta = abs(far - frr)
            prev_delta = abs(memory[0] - memory[1]) if memory else float('inf')
            if delta <= prev_delta:
                best_eer = (far + frr) / 2
                best_thresh = score
            else:
                best_eer = (memory[0] + memory[1]) / 2
                best_thresh = memory[2]
            return best_eer, best_thresh
        memory = [far, frr, score]
    return best_eer, best_thresh

#############################
# Data loader
#############################
class FeatureDataset(Dataset):
    def __init__(self, list_file):
        self.samples = [line.strip().split() 
                        for line in open(list_file) 
                        if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        df = pd.read_csv(path, encoding='utf-16')
        feats = torch.from_numpy(df.values.astype(np.float32))
        return feats, int(lbl)

def collate_fn_eval(batch):
    feats, labels = zip(*batch)
    seq_lens = [f.size(0) for f in feats]
    max_len = max(seq_lens)
    B, D = len(feats), feats[0].size(1)
    padded = torch.zeros(B, max_len, D)
    for i,f in enumerate(feats):
        padded[i, :f.size(0), :] = f
    return padded, torch.tensor(labels), seq_lens

def get_atten_mask(seq_lens, B):
    M = max(seq_lens)
    mask = torch.zeros(B, M, M, dtype=torch.bool)
    for i, L in enumerate(seq_lens):
        mask[i, :L, :L] = True
    return mask

#############################
# Configuration
#############################
MODEL_PATH = "/home/teaching/Transformermd_mfcc_22_04_4.ckpt"
TEST_LIST  = "/home/teaching/conformer_mfcc/unseentest/lists/all_list.txt"
MODEL_TYPE = "Transformer"   # or "Transformer" / "Conformer"
DIM, N_LANG = 39, 12     # adjust as needed
DEVICE_ID = 0
BATCH_SIZE= 1

#############################
# Run evaluation
#############################
def main():
    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

    # 1) init model
    if MODEL_TYPE == "XSA_E2E":
        model = X_Transformer_E2E_LID(
            input_dim=DIM, feat_dim=64, d_k=64, d_v=64, d_ff=2048,
            n_heads=4, dropout=0.1, n_lang=N_LANG,
            max_seq_len=20000, device=device
        )
    elif MODEL_TYPE == "Transformer":
        model = Transformer_E2E_LID(
            input_dim=DIM, feat_dim=64, d_k=64, d_v=64, d_ff=2048,
            n_heads=8, dropout=0.1, n_lang=N_LANG,
            max_seq_len=20000, device=device
        )
    else:
        model = Conformer(
            input_dim=DIM, feat_dim=64, d_k=64, d_v=64, n_heads=8,
            d_ff=2048, max_len=20000, dropout=0.1,
            n_lang=N_LANG, device=device
        )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # 2) data loader
    dataset = FeatureDataset(TEST_LIST)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn_eval)

    # 3) inference & metrics
    correct = 0
    total   = len(dataset)
    pairs_2 = []

    with torch.no_grad():
        for feats, labels, seq_lens in tqdm(loader, desc="Eval"):
            feats = feats.to(device)
            B = feats.size(0)
            mask = get_atten_mask(seq_lens, B).to(device)

            logits = model(feats, seq_lens, mask)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            correct += (preds.cpu() == labels).sum().item()

            # collect for EER
            for i in range(B):
                true_lbl = labels[i].item()
                for l in range(N_LANG):
                    score = probs[i,l].item()
                    pairs_2.append((score, 1 if l==true_lbl else 0))

    # 4) compute & print
    accuracy = correct / total
    eer, thresh = compute_eer(pairs_2, total, total*(N_LANG-1))

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"EER:      {eer*100:.2f}%  @ threshold {thresh:.4f}")

if __name__ == "__main__":
    main()

