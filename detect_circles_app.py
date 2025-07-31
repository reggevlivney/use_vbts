#!/usr/bin/env python
"""
tag_normals_interactive.py  (click tag + save centre/radius JSON)

Per saved frame you now get:
    normals_tagged/<stem>.npy   – float32 [Hc,Wc,3] unit normals
    normals_tagged/<stem>.json  – {"cx":…, "cy":…, "R":…}

Controls
  • left-click centre, left-click rim
  • r  reset | Enter/Space accept | Esc quit
Edit CROP_* and CROP_MARGIN as needed.
"""

import cv2, numpy as np, pathlib, math, sys, json

# ───────── USER SETTINGS ─────────────────────────────────────────────
CROP_TOP, CROP_BOTTOM = 0, 0
CROP_LEFT, CROP_RIGHT = 0, 0
CROP_MARGIN           = 5
R_MAX                 = 95
ROOT_DIR              = pathlib.Path("dataset")
IMAGE_SUBDIR          = "images_digit"
NORMAL_SUBDIR         = "normals_tagged_digit"
EMPTY_NAME            = "empty.jpg"
# --------------------------------------------------------------------

image_dir  = ROOT_DIR / IMAGE_SUBDIR
normal_dir = ROOT_DIR / NORMAL_SUBDIR
normal_dir.mkdir(parents=True, exist_ok=True)

EXTS      = {".jpg", ".jpeg", ".png"}
WIN       = "GelSight Tag  |  click centre + rim   (r=reset  Enter=save  Esc=quit)"
clicks    = []      # shared list, cleared each frame

# ───────── helper functions ─────────────────────────────────────────
def crop(img):
    h, w = img.shape[:2]
    return img[
        CROP_TOP : h - CROP_BOTTOM if CROP_BOTTOM else h,
        CROP_LEFT: w - CROP_RIGHT  if CROP_RIGHT  else w
    ]

def load_and_crop(stem):
    for ext in EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            im = cv2.imread(str(p))
            return crop(im) if im is not None else None
    return None

def preview(img):
    vis = img.copy()
    if clicks:
        cx, cy = clicks[0]
        cv2.drawMarker(vis, (cx, cy), (0,0,255), cv2.MARKER_CROSS, 20, 2)
    if len(clicks) == 2:
        cx, cy = clicks[0]; rx, ry = clicks[1]
        R = int(round(math.hypot(rx-cx, ry-cy)))
        cv2.circle(vis, (cx, cy), R, (0,255,0), 2)
    return vis

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))

# ───────── gather frames needing annotation ─────────────────────────
stems = sorted({p.stem for p in image_dir.iterdir()
                if p.suffix.lower() in EXTS
                and p.name != EMPTY_NAME
                and not (normal_dir/f"{p.stem}.npy").exists()})

if not stems:
    print("Nothing to tag — all frames already processed.")
    sys.exit()

print(f"{len(stems)} frames to tag.\n"
      "• click centre, click rim  |  r=reset  |  Enter/Space=save  |  Esc=quit")

# ───────── make GUI window once and attach callback ─────────────────
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.imshow(WIN, np.zeros((50,50,3), np.uint8))   # placeholder so window exists
cv2.setMouseCallback(WIN, on_mouse)

for stem in stems:
    img = load_and_crop(stem)
    if img is None:
        print(f"[WARN] cannot read {stem}.* – skipped")
        continue

    H, W = img.shape[:2]
    clicks.clear()

    while True:
        cv2.imshow(WIN, preview(img))
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                        # Esc
            print("User aborted.")
            cv2.destroyAllWindows()
            sys.exit()
        if key in (ord('r'), ord('R')):      # reset clicks
            clicks.clear()
        if key in (13, 32) and len(clicks) == 2:   # Enter or Space
            break

    cx, cy = clicks[0]; rx, ry = clicks[1]
    R = float(math.hypot(rx - cx, ry - cy))
    print(f"{stem}: centre=({cx},{cy})  R={R:.1f}")

    # compute normals
    x0 = int(max(0, cx - R_MAX - CROP_MARGIN));  x1 = int(min(W, cx + R_MAX + CROP_MARGIN))
    y0 = int(max(0, cy - R_MAX - CROP_MARGIN));  y1 = int(min(H, cy + R_MAX + CROP_MARGIN))
    xs, ys = np.meshgrid(np.arange(x0,x1), np.arange(y0,y1))
    dx, dy = xs - cx, ys - cy
    r2     = dx*dx + dy*dy
    inside = r2 < R*R
    nz = np.sqrt(np.clip(R_MAX*R_MAX - r2, 0, None))
    nx, ny = -dx, -dy
    norm   = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-8
    normals = np.zeros((y1-y0, x1-x0, 3), np.float32)
    normals[...,2] = 1
    normals[inside] = np.stack([nx[inside], ny[inside], nz[inside]], -1) / norm[inside,None]

    # save normal map + metadata
    np.save(normal_dir/f"{stem}.npy", normals)
    meta = {"cx": int(cx), "cy": int(cy), "R": R}
    with open(normal_dir/f"{stem}.json", "w") as jf:
        json.dump(meta, jf)

cv2.destroyAllWindows()
print("✓ Tagged normals & metadata saved to", normal_dir)
