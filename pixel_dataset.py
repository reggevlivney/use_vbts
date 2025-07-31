# datasets/gelsight_pixels_from_npy.py
import time
from pathlib import Path
import cv2, numpy as np, torch, random, math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import json
import tqdm

class GelSightPixelDataset(Dataset):
    """
    Pixel-level dataset that reads cached ΔRGB frames and pre-computed
    normal maps (*.npy) from dataset/normals_tagged.
    Each epoch:  K = pixels_per_img random pixels per image.
                 50 % from inside the labelled disc, 50 % anywhere.
    Feature: [dR, dG, dB, x̃, ỹ]      shape (5,)
    Target : [nx, ny, nz] (unit)       shape (3,)
    """
    def __init__(self, root="dataset", split=None,
                 pixels_per_img=4096, scale=0.1, permutation=None):
        root      = Path(root)
        imgdir    = root / "images_digit"
        tagdir    = root / "normals_tagged_digit"

        # ---- reference (empty) frame --------------------------------
        # ref = cv2.imread(str(imgdir/"empty.jpg"))[..., ::-1]  # BGR→RGB
        ref = cv2.imread(str(imgdir/"empty.jpg"))  # BGR→RGB
        if scale != 1.0:
            ref = cv2.resize(ref, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        self.ref = ref.astype(np.float32) / 255.

        # ---- collect stems (real & empty) ---------------------------
        stems = sorted(p.stem for p in tagdir.glob("*.npy"))
        if permutation is not None:
            stems = [stems[i] for i in permutation]
        train_cut = int(0.8*len(stems))
        if split == "train":
            self.stems = stems[:train_cut]
        elif split == "val":
            self.stems = stems[train_cut:]
        else:
            self.stems = stems

        # ---- load ΔRGB frames + normal maps -------------------------
        self.drgb  = {}      # stem -> H×W×3 float
        self.norms = {}      # stem -> H×W×3 float (unit)
        self.meta  = {}      # stem -> (cx,cy,R,H,W)
        for stem in tqdm.tqdm(self.stems, desc=f"[{split}] caching", ncols=80):
            # JPEG: for '0001e' reuse '0001.jpg'
            jpg_stem = stem[:-1] if stem.endswith('e') else stem
            jpg_path = imgdir / f"{jpg_stem}.jpg"
            if not jpg_path.exists():
                print(f"⚠  skip {stem}: missing {jpg_path.name}")
                continue

            # img = cv2.imread(str(jpg_path))[..., ::-1]
            img = cv2.imread(str(jpg_path))
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            drgb = img.astype(np.float32)/255.0 - self.ref
            self.drgb[stem] = drgb

            nmap = np.load(tagdir/f"{stem}.npy")
            if scale != 1.0:
                chans = [cv2.resize(nmap[...,c], (drgb.shape[1], drgb.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
                         for c in range(3)]
                nmap = np.stack(chans, -1)
            # replace zero normals by flat unit [0,0,1]
            zero_mask = (nmap == 0).all(-1)
            nmap[zero_mask] = np.array([0,0,1], np.float32)
            self.norms[stem] = nmap.astype(np.float32)

            # meta for disc sampling
            with open(tagdir/f"{stem[:-1] if stem.endswith('e') else stem}.json") as f:
                m = json.load(f)
            self.meta[stem] = (m["cx"]*scale, m["cy"]*scale,
                               m["R"]*scale, drgb.shape[0], drgb.shape[1])

        # ---- build index table --------------------------------------
        self.K   = pixels_per_img
        self.map = [(s,i) for s in self.stems for i in range(self.K)]

        print(f"[DatasetNPY] {split}: {len(self.stems)} images × "
              f"{self.K} = {len(self)} samples")

    def __len__(self): return len(self.map)

    @staticmethod
    def _rand_inside(cx, cy, R, H, W):
        while True:
            r   = math.sqrt(random.random()) * R
            ang = random.random()*2*math.pi
            x   = int(cx + r*math.cos(ang))
            y   = int(cy + r*math.sin(ang))
            if 0 <= x < W and 0 <= y < H:
                return x, y

    def __getitem__(self, idx):
        stem, _ = self.map[idx]
        img  = self.drgb[stem]          # H×W×3
        nmap = self.norms[stem]
        cx,cy,R,H,W = self.meta[stem]

        # 50 % from disc (if R>0 and not an 'e' stem), else anywhere
        if (R > 0) and (not stem.endswith('e')) and random.random() < 0.5:
            x,y = self._rand_inside(cx, cy, R, H, W)
        else:
            x = random.randint(0, W-1); y = random.randint(0, H-1)

        dR,dG,dB = img[y,x]
        feat = np.array([dR, dG, dB,
                         2*x/W - 1,
                         2*y/H - 1], np.float32)
        target = nmap[y,x]

        return torch.from_numpy(feat), torch.from_numpy(target)


    def get_pixel_values(self,y,x):
        drgbvals = np.array([self.drgb[stem][y,x] for stem in self.stems])
        nmapvals = np.array([self.norms[stem][y,x] for stem in self.stems])
        return drgbvals, nmapvals

if __name__ == "__main__":
    g = GelSightPixelDataset(scale=1)
    drgbvals, nmapvals = g.get_pixel_values(120,160)
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(range(182),255*drgbvals[:,1])
    plt.subplot(2, 1, 2)
    plt.scatter(range(182),nmapvals[:,1])
    plt.show()

    stem, _ = g.map[162]
    img = g.drgb[stem]  # H×W×3
    nmap = g.norms[stem]

    plt.figure()
    plt.imshow((255*img+128).astype(np.uint8))
    plt.show()
    pass
    plt.figure()
    plt.imshow(nmap)
    plt.show()
    plt.pause(1)
    pass

