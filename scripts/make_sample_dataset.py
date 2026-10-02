"""สร้าง dataset/sample ขนาดเล็กจาก dataset/tomato เพื่อ commit ขึ้น GitHub ให้ CI ใช้

dataset/tomato เต็มมีขนาดราว 1 GB จึงไม่ควรขึ้น Git ส่วน sample นี้มีแค่ไม่กี่ MB

    python scripts/make_sample_dataset.py            # ค่าเริ่มต้น 40/10/10 ภาพต่อคลาส
    python scripts/make_sample_dataset.py 60 15 15   # กำหนดจำนวน train val test เอง
"""

import shutil
import sys

import numpy as np

from common import IMAGE_EXTS, ROOT, SEED, SPLITS, list_classes

SRC = ROOT / "dataset" / "tomato"
DST = ROOT / "dataset" / "sample"


def main(per_split: list[int]):
    rng = np.random.default_rng(SEED)
    if DST.exists():
        shutil.rmtree(DST)
    total = 0
    for split, n in zip(SPLITS, per_split):
        for name in list_classes(SRC / split):
            class_dir = SRC / split / name
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            out = DST / split / class_dir.name
            out.mkdir(parents=True)
            for i in rng.choice(len(files), min(n, len(files)), replace=False):
                shutil.copy2(files[i], out / files[i].name)
                total += 1
    print(f"Copied {total} images to {DST}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [40, 10, 10]
    main(args)
