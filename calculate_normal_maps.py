from pathlib import Path

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as pp

if __name__ == "__main__":
    imH = 640
    imW = 480
    R_MAX = 95

    root = Path('dataset')
    tagdir    = root / "normals_tagged_digit"

    xMat, yMat = np.meshgrid(range(imH),range(imW))
    base_norm_mat = np.zeros([imW,imH,3])
    # base_norm_mat[...,2] = 1

    stems = sorted(p.stem for p in tagdir.glob("*.json"))
    for stem in stems:
        with open(tagdir / f"{stem}.json") as f:
            js = json.load(f)
            cx = js['cx']
            cy = js['cy']
            R  = js['R']
            print(stem)
            norm_xy = np.sqrt((xMat-cx)**2 + (yMat-cy)**2)/R_MAX
            mask_xy = (norm_xy < R/R_MAX).astype(np.double)
            angle_xy = np.arctan2(yMat-cy,xMat-cx)
            base_norm_mat[..., 0] = mask_xy * norm_xy * np.cos(angle_xy)
            base_norm_mat[..., 1] = mask_xy * norm_xy * np.sin(angle_xy)
            base_norm_mat[..., 2] = np.sqrt(1 - mask_xy * norm_xy)

            # cv2.imshow("norms",cv2.resize(128*base_norm_mat + 128,(328,246)).astype(np.uint8))
            # cv2.waitKey(1)
            np.save(tagdir / f"{stem}.npy",base_norm_mat)
            pass


