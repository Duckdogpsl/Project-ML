"""ขั้นที่ 2: แปลงภาพทุกภาพเป็นเวกเตอร์ฟีเจอร์ แล้วบันทึกเป็น processed_data/<split>.npz

แต่ละไฟล์ .npz มี
  X     : ฟีเจอร์ ขนาด (จำนวนภาพ, จำนวนฟีเจอร์)
  y     : ชื่อคลาส (เช่น 'Leaf_Mold')
  paths : path ของภาพต้นฉบับ (ไว้ตรวจย้อนหลัง)

ตัดภาพซ้ำออก (DEDUP=1 เป็นค่าเริ่มต้น): dataset นี้มีภาพเดียวกันอยู่ทั้งใน train และ val/test
หลายพันภาพ ถ้าไม่ตัดออก ผล accuracy บน val/test จะสูงเกินจริงเพราะโมเดลเคยเห็นภาพนั้นแล้ว
ไล่ตามลำดับ train -> val -> test แล้วเก็บเฉพาะภาพที่ยังไม่เคยเจอ
"""

import hashlib
import json
import os
import time

import mlflow
import numpy as np

from common import (
    DATA_DIR,
    HIST_BINS,
    IMG_SIZE,
    MAX_PER_CLASS,
    PROCESSED_DIR,
    SPLITS,
    extract_features_from_path,
    list_classes,
    list_images,
    setup_mlflow,
    short_label,
)

DEDUP = os.getenv("DEDUP", "1") == "1"


def preprocess():
    setup_mlflow()
    PROCESSED_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_params(
            {
                "data_dir": str(DATA_DIR),
                "img_size": IMG_SIZE,
                "hist_bins": HIST_BINS,
                "features": "hsv_hist+hog",
                "max_per_class": MAX_PER_CLASS,
                "dedup": DEDUP,
            }
        )

        classes = [short_label(c) for c in list_classes(DATA_DIR / "train")]
        start = time.time()
        seen: set[str] = set()

        for split in SPLITS:
            X, y, paths = [], [], []
            dropped = 0
            for class_dir in list_classes(DATA_DIR / split):
                label = short_label(class_dir)
                for f in list_images(DATA_DIR / split / class_dir):
                    if DEDUP:
                        digest = hashlib.md5(f.read_bytes()).hexdigest()
                        if digest in seen:
                            dropped += 1
                            continue
                        seen.add(digest)
                    X.append(extract_features_from_path(f))
                    y.append(label)
                    paths.append(str(f.relative_to(DATA_DIR)))
            X = np.stack(X)
            np.savez_compressed(
                PROCESSED_DIR / f"{split}.npz", X=X, y=np.array(y), paths=np.array(paths)
            )
            mlflow.log_metric(f"num_samples_{split}", len(y))
            mlflow.log_metric(f"duplicates_dropped_{split}", dropped)
            print(f"{split:>5}: {X.shape[0]} images (dropped {dropped} duplicates) -> features {X.shape}")

        mlflow.log_metric("num_features", X.shape[1])
        mlflow.log_metric("preprocess_seconds", round(time.time() - start, 1))
        (PROCESSED_DIR / "classes.json").write_text(json.dumps(classes, indent=2))
        mlflow.log_artifact(str(PROCESSED_DIR / "classes.json"))
        print(f"Saved to {PROCESSED_DIR} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    preprocess()
