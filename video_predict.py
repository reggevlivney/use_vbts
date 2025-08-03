#!/usr/bin/env python
# -*- coding: utf-8 -*-
# video_height_scaled.py ----------------------------------------------
"""
Small-resolution height-map video generator.
 * Uses ΔRGB 5-input MLP (32-32-32 tanh, FP16)
 * Poisson FFT runs on GPU
 * Whole pipeline runs at SCALE × original size; output video is small.
"""
import time

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.fftpack import fftfreq
import matplotlib
from matplotlib import cm
# import visualize3d
from pixel_mlp import PixelMLP32Tanh
import keyboard
from queue import Queue

# ───────── configuration ─────────────────────────────────────────────
ROOT         = Path("dataset")
VIDEO_IN     = ROOT/"video"/"sensor_feed_3.mp4"
VIDEO_OUT    = ROOT/"video"/"video_output.mp4"
EMPTY_IMG    = ROOT/"images"/"empty.jpg"
MODEL_PTH    = "pixel_mlp_normals.pth"
VIDEO_SOURCE = "file" # "camera","file"
CAMERA_ID    = 0

print(f'Input from {VIDEO_SOURCE}. Checking CUDA availability...')
SCALE       = 1 # if VIDEO_SOURCE=="file" else 0.1   # ← SAME scale you used in training
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_PIX   = 65536
NZ_THRESH   = 0.2
MAG_THRESH  = 0.02
MORPH_K     = 9

viridis = cm.get_cmap("viridis")
assert VIDEO_IN.exists() and EMPTY_IMG.exists() and Path(MODEL_PTH).exists()

# ───────── GPU Poisson helpers ───────────────────────────────────────
def make_freq_grid(H, W, device):
    fy, fx = torch.meshgrid(torch.fft.fftfreq(H, device=device),
                            torch.fft.fftfreq(W, device=device),
                            indexing='ij')
    denom = fx*fx + fy*fy
    denom[0,0] = 1.0
    return fx, fy, denom

def poisson_height_gpu(p, q, fx, fy, denom):
    Z = ((1j*fx)*torch.fft.fft2(p) + (1j*fy)*torch.fft.fft2(q)) / denom
    h = torch.fft.ifft2(Z).real
    h -= h.amin(); h /= (h.amax() - h.amin() + 1e-8)
    return h

def height_to_bgr(h_np):
    rgb = (viridis(h_np)[..., :3]*255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ───────── open video streams ────────────────────────────────────────
# @line_profiler.profile
def main():
    info_string = str(VIDEO_IN) if VIDEO_SOURCE=="file" else f'Camera ID:{CAMERA_ID}'
    print(f'Opening input stream ({info_string})...')
    cap = cv2.VideoCapture(str(VIDEO_IN) if VIDEO_SOURCE=="file" else CAMERA_ID)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError("Cannot open video")
    fps   = cap.get(cv2.CAP_PROP_FPS) * (0.3 if VIDEO_SOURCE=='camera' else 1)
    Wfull = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hfull = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # scaled dimensions
    Ws = int(round(Wfull * SCALE))
    Hs = int(round(Hfull * SCALE))

    print('Opening output video stream...')
    four = cv2.VideoWriter_fourcc(*"mp4v")
    Path(VIDEO_OUT).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(VIDEO_OUT), four, fps, (Ws, Hs))

    # ───────── load reference & model ────────────────────────────────────
    # if VIDEO_SOURCE == "file":
    #     ref_full = cv2.imread(str(EMPTY_IMG))[..., ::-1].astype(np.float32)/255.
    # else:
    print('Getting reference image - do not touch sensor.')
    res, ref_full = cap.read()
    # ref_full = ref_full[..., ::-1].astype(np.float32)/255
    ref_full = ref_full.astype(np.float32)/255

    ref = cv2.resize(ref_full, (Ws, Hs), interpolation=cv2.INTER_AREA)

    print('Loading neural net...')
    net = PixelMLP32Tanh(in_dim=5,out_dim=3,layer_size=64).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
    net.eval()

    print('Mathematical preparations...')
    # frequency grid for Poisson FFT
    fx, fy, denom = make_freq_grid(Hs, Ws, DEVICE)

    print(f"[info] frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, "
          f"orig {Wfull}×{Hfull}  →  scaled {Ws}×{Hs}")

    # pre-compute x̃, ỹ grids (on CPU, float32)
    ys, xs = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing='ij')
    xs = xs.astype(np.float32); ys = ys.astype(np.float32)

    # ───────── per-frame loop ────────────────────────────────────────────
    print('Opening 3D visualizer...')
    frame_idx = 0
    # vis3d = visualize3d.Visualize3D(Ws,Hs,'',None)
    print('Let\'s go! Press q to end run.')

    # Init queue
    Nqueue = 3
    bgr_queue = Queue()
    for i in range(Nqueue):
        ret, bgr = cap.read()
        bgr_queue.put(bgr/Nqueue)

    with torch.no_grad():
        while True:
            ret, bgr_cap = cap.read()
            if not ret: break
            bgr_queue.get()
            bgr_queue.put(bgr_cap/Nqueue)
            bgr = np.sum(np.array(bgr_queue.queue),0)
            #if keyboard.is_pressed('q') or not ret: break
            frame_idx += 1

            # -------- ΔRGB, resize ----------
            # rgb = bgr[..., ::-1].astype(np.float32)/255.
            rgb = bgr.astype(np.float32)/255.
            rgb_s = cv2.resize(rgb, (Ws, Hs), interpolation=cv2.INTER_AREA) if VIDEO_SOURCE=="camera" else rgb
            d = rgb_s - ref

            # -------- feature tensor (Nqueue×5) ---
            feat = np.stack([ d[...,0], d[...,1], d[...,2],
                              2*xs/Ws-1, 2*ys/Hs-1 ], -1)\
                   .reshape(-1,5).astype(np.float32)

            # -------- MLP inference ----------
            t2 = time.time()
            preds=[]
            for i in range(0, feat.shape[0], BATCH_PIX):
                chunk = torch.from_numpy(feat[i:i+BATCH_PIX]).to(DEVICE)
                preds.append(net(chunk).float().cpu().numpy())
            normals = np.concatenate(preds,0).reshape(Hs,Ws,3)
            print(time.time() - t2)

            # -------- mask & slopes ----------
            nz  = normals[...,2]
            mag = np.linalg.norm(d, axis=-1)
            mask = (mag > MAG_THRESH) & (nz >= NZ_THRESH)
            SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_K,MORPH_K))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, SE)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,SE)

            nx, ny = normals[...,0], normals[...,1]
            p = np.zeros_like(nz, dtype=np.float32)
            q = np.zeros_like(nz, dtype=np.float32)
            inside = mask.astype(bool)
            if inside.any():
                p[inside] = -nx[inside]/(nz[inside]+1e-6)
                q[inside] = -ny[inside]/(nz[inside]+1e-6)
                np.clip(p,-10,10,out=p); np.clip(q,-10,10,out=q)
            p -= p.mean(); q -= q.mean()

            # -------- Poisson FFT on GPU -----
            p_t = torch.from_numpy(p).to(DEVICE)
            q_t = torch.from_numpy(q).to(DEVICE)
            h_t = poisson_height_gpu(-p_t, -q_t, fx, fy, denom)
            height = h_t.cpu().numpy()            # Hs×Ws float32 0-1
            if frame_idx == 1:
                height0 = height
            height = height - height0

            # -------- write frame ------------
            out.write(height_to_bgr(height))
            # vis3d.update(500*height)
            disp_normals = normals[...,::-1].copy() + 1
            disp_normals[...,0] = 0
            cv2.imshow("Image",rgb_s)
            if frame_idx%50 == 0:
                print(f"  processed {frame_idx} frames", flush=True)
            # if frame_idx==100:
            #     break
    cap.release(); out.release()
    print("[done] small height video saved:", VIDEO_OUT)

if __name__ == "__main__":
    main()
