#!/usr/bin/env python
"""
Rename all image files in <folder> to 0001.ext, 0002.ext, …
Keeps the original extension (.png, .jpg, …) and orders files
alphabetically before renaming.

Usage:
    python rename_images.py path/to/images
"""
import sys, shutil
from pathlib import Path

def main(folder):
    folder = Path(folder)
    imgs   = sorted([p for p in folder.iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}])

    if not imgs:
        print("No images found – check the path or extensions.")
        return

    # --- safety first --------------------------------------------------------
    bak = folder.with_name(folder.name + "_backup")
    if not bak.exists():
        print(f"Creating backup copy → {bak}")
        shutil.copytree(folder, bak)
    # -------------------------------------------------------------------------

    for i, f in enumerate(imgs, 1):
        new_name = f"{i:04d}{f.suffix.lower()}"
        print(f"{f.name:>30}  →  {new_name}")
        f.rename(f.with_name(new_name))

    print("\nDone ✔")

if __name__ == "__main__":
    main('dataset\\images_digit')
