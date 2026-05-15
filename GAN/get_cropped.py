# preprocess_face_crops.py
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
SRC_DIR   = Path("../../deepfake_data/wiki")
DST_DIR   = Path("../../deepfake_data/wiki_cropped")

MARGIN          = 0.45
SCALE_FACTOR    = 1.1
MIN_NEIGHBORS   = 5
MIN_FACE_SIZE   = 20
JPEG_QUALITY    = 95

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# ──────────────────────────────────────────────────────────────────────────────


def _center_square_crop(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _expand_to_square_box(x, y, w, h, img_w, img_h, margin=0.45):
    cx   = x + w / 2
    cy   = y + h / 2
    side = max(w, h) * (1.0 + margin)

    left   = int(round(cx - side / 2))
    top    = int(round(cy - side / 2))
    right  = int(round(cx + side / 2))
    bottom = int(round(cy + side / 2))

    left   = max(0, left)
    top    = max(0, top)
    right  = min(img_w, right)
    bottom = min(img_h, bottom)

    bw = right - left
    bh = bottom - top
    if bw != bh:
        side2 = min(max(bw, bh), img_w, img_h)
        cx2   = (left + right)  // 2
        cy2   = (top  + bottom) // 2
        left  = max(0, min(img_w - side2, cx2 - side2 // 2))
        top   = max(0, min(img_h - side2, cy2 - side2 // 2))
        right  = left + side2
        bottom = top  + side2

    return left, top, right, bottom


def crop_image(img, cascade):
    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        l, t, r, b = _expand_to_square_box(x, y, w, h, img.size[0], img.size[1], margin=MARGIN)
        return img.crop((l, t, r, b))
    else:
        return _center_square_crop(img)


def main():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    assert not cascade.empty(), "Haar cascade failed to load."

    # Collect all fold dirs in sorted order (mirrors wiki structure exactly)
    fold_dirs = sorted(d for d in SRC_DIR.iterdir() if d.is_dir())
    print(f"Found {len(fold_dirs)} folds in {SRC_DIR}")

    saved = skipped = errors = 0

    for fold_dir in tqdm(fold_dirs, desc="Folds", unit="fold"):
        dst_fold = DST_DIR / fold_dir.name
        dst_fold.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in fold_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for src_path in tqdm(files, desc=fold_dir.name, unit="img", leave=False):
            dst_path = (dst_fold / src_path.name).with_suffix(".jpg")

            if dst_path.exists():
                skipped += 1
                continue

            try:
                img     = Image.open(src_path).convert("RGB")
                cropped = crop_image(img, cascade)
                cropped.save(dst_path, "JPEG", quality=JPEG_QUALITY)
                saved += 1
            except Exception as e:
                tqdm.write(f"ERROR {src_path}: {e}")
                errors += 1

    print(f"\nDone — saved: {saved}, skipped: {skipped}, errors: {errors}")
    print(f"Output: {DST_DIR}")


if __name__ == "__main__":
    main()