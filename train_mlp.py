#!/usr/bin/env python
import time

import numpy as np
import torch, torch.nn.functional as F
from mkl_random.mklrand import permutation
from torch.utils.data import DataLoader
from pixel_dataset import GelSightPixelDataset
from pixel_mlp import PixelMLP32Tanh
import tqdm
import random
import multiprocessing as mp
from pathlib import Path

DEVICE = "cpu" # cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 4096
EPOCHS = 10
LR     = 1e-3

# def cos_loss(p,g): return (1 - (p*g).sum(-1)).mean()
def cos_loss(pred, gt):
    mask = gt.norm(dim=-1) > 0.5
    return (1 - (pred[mask]*gt[mask]).sum(-1)).mean()

def main():
    print('Loading datasets...')
    root = Path('dataset')
    tagdir = root / "normals_tagged_digit"
    num_tags = len([p.stem for p in tagdir.glob("*.npy")])
    permutation = np.random.permutation(range(num_tags))
    train_ds = GelSightPixelDataset(split="train", scale=1, permutation=permutation)
    val_ds   = GelSightPixelDataset(split="val",   scale=1,
                                        pixels_per_img=4096, permutation=permutation)
    print('Creating Data Loader...')
    train_ld = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   BATCH, shuffle=False,num_workers=0)

    model = PixelMLP32Tanh(in_dim=5, out_dim=3, layer_size=64).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        model.train(); tot=0
        for f,g in tqdm.tqdm(train_ld, desc=f"Ep{ep} train", leave=True):
            f,g = f.to(DEVICE), g.to(DEVICE)
            loss = cos_loss(model(f), g)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*f.size(0)
        tr = tot/len(train_ds)

        model.eval(); tot=0
        with torch.no_grad():
            for f,g in tqdm.tqdm(val_ld, desc=f"Ep{ep} val", leave=True):
                f,g = f.to(DEVICE), g.to(DEVICE)
                tot += cos_loss(model(f), g).item()*f.size(0)
        val = tot/len(val_ds)
        print(f"Ep{ep:02d}  train {tr:.4f} | val {val:.4f}\n")
        time.sleep(0.01)

    torch.save(model.state_dict(), "pixel_mlp_normals_digit.pth")
    print("✓ saved  pixel_mlp_normals_digit.pth")

if __name__ == "__main__":
    mp.freeze_support()
    main()