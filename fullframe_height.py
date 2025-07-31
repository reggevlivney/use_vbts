#!/usr/bin/env python
# fullframe_multiheight_diff5_small.py
import cv2, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.fft import fft2, ifft2, fftfreq
from pixel_mlp import PixelMLP32Tanh

# ───────── parameters ────────────────────────────────────────────────
ROOT         = Path("dataset")
IMG_PATH     = sorted((ROOT/"images").glob("*.jpg"))[15]
EMPTY_PATH   = ROOT/"images"/"empty.jpg"
MODEL_PTH    = "pixel_mlp_normals.pth"

SCALE        = 0.10          # same scale used for training
BATCH_PIXELS = 65536
NZ_THRESH    = 0.2
MORPH_K      = 9
MAG_THRESH   = 0.02
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def poisson_height(p, q):
    H, W = p.shape
    fy, fx = np.meshgrid(fftfreq(H), fftfreq(W), indexing='ij')  # ← one call per dim
    P, Q  = fft2(p), fft2(q)
    denom = fx**2 + fy**2
    denom[0, 0] = 1.0                         # avoid divide-by-zero at DC
    Z = ((1j * fx) * P + (1j * fy) * Q) / denom
    h = np.real(ifft2(Z))
    h -= h.min()
    h /= h.ptp() + 1e-8                       # normalise to 0-1
    return h.astype(np.float32)

# ───────── load & downscale frames ───────────────────────────────────
frame_full = cv2.imread(str(IMG_PATH))[..., ::-1].astype(np.float32)/255.
ref_full   = cv2.imread(str(EMPTY_PATH))[..., ::-1].astype(np.float32)/255.
Hs = int(round(frame_full.shape[0]*SCALE))
Ws = int(round(frame_full.shape[1]*SCALE))
frame = cv2.resize(frame_full, (Ws, Hs), interpolation=cv2.INTER_AREA)
ref   = cv2.resize(ref_full,   (Ws, Hs), interpolation=cv2.INTER_AREA)

# ───────── feature tensor ────────────────────────────────────────────
d = frame - ref
ys,xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing='ij')
feat = np.stack([ d[...,0], d[...,1], d[...,2],
                  2*xs/Ws-1, 2*ys/Hs-1 ], -1)\
       .reshape(-1,5).astype(np.float32)

# ───────── inference ────────────────────────────────────────────────
net = PixelMLP32Tanh(in_dim=5,out_dim=3,layer_size=64).to(DEVICE)
net.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
net.eval()

parts=[]
with torch.no_grad():
    for i in tqdm(range(0, feat.shape[0], BATCH_PIXELS)):
        chunk = torch.from_numpy(feat[i:i+BATCH_PIXELS]).to(DEVICE)
        parts.append(net(chunk).cpu().numpy())
normals = np.concatenate(parts,0).reshape(Hs,Ws,3)

# ───────── mask, slopes, Poisson ─────────────────────────────────────
nz = normals[...,2]
mag = np.linalg.norm(d, axis=-1)
mask = ((mag > MAG_THRESH) & (nz >= NZ_THRESH)).astype(np.uint8)
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_K,MORPH_K))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, SE)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,SE)

nx,ny = normals[...,0], normals[...,1]
p = np.zeros_like(nz); q = np.zeros_like(nz)
inside = mask.astype(bool)
p[inside] = -nx[inside]/(nz[inside]+1e-6)
q[inside] = -ny[inside]/(nz[inside]+1e-6)
p = np.clip(p,-10,10); q = np.clip(q,-10,10)
p -= p.mean(); q -= q.mean()
# height = poisson_height(p,q)
height = poisson_height(-p,q)
# height = poisson_height(q,p)
# height = poisson_height(-q,p)
normals_disp = normals.copy()

# ───────── save / show  (small size) ─────────────────────────────────
cv2.imwrite("normals_small.png",
            cv2.cvtColor(((normals+1)/2*255).astype(np.uint8),
                         cv2.COLOR_RGB2BGR))
cv2.imwrite("height_small.png", (height*255).astype(np.uint8))

plt.figure(figsize=(12,3))
plt.subplot(131); plt.title("Scaled RGB");     plt.imshow(frame);        plt.axis("off")
# plt.subplot(132); plt.title("Normals");        plt.imshow((normals+1)/2);plt.axis("off")
plt.subplot(132); plt.title("Normals");        plt.imshow(normals_disp[...,2]);plt.axis("off")
plt.subplot(133); plt.title("Height");         plt.imshow(height,cmap='viridis'); plt.axis("off")
plt.tight_layout(); plt.show()
