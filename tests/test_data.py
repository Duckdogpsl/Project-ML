"""ทดสอบโครงสร้าง dataset และฟังก์ชันสกัดฟีเจอร์"""

import numpy as np
import pytest
from PIL import Image

from common import CLASSES, DATA_DIR, SPLITS, extract_features, list_classes, short_label


def random_image(size=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))


def test_short_label():
    assert short_label("Tomato__Tomato_Leaf_Mold") == "Leaf_Mold"
    assert short_label("Tomato__Tomato_Yellow_Leaf_Curl_Virus") == "Yellow_Leaf_Curl_Virus"


def test_features_shape_and_dtype():
    f = extract_features(random_image())
    assert f.ndim == 1
    assert f.dtype == np.float32
    assert np.isfinite(f).all()


def test_features_independent_of_input_size():
    """ภาพต่างขนาดต้องได้ฟีเจอร์ยาวเท่ากัน ไม่งั้นโมเดลรับไม่ได้"""
    a = extract_features(random_image((256, 256)))
    b = extract_features(random_image((120, 300)))
    assert a.shape == b.shape


def test_features_deterministic():
    img = random_image()
    np.testing.assert_array_equal(extract_features(img), extract_features(img))


def test_color_histogram_is_normalized():
    f = extract_features(random_image())
    assert f[:512].sum() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not DATA_DIR.exists(), reason=f"ไม่พบ dataset ที่ {DATA_DIR}")
def test_dataset_structure():
    reference = list_classes(DATA_DIR / "train")
    if CLASSES is not None:
        assert reference == sorted(CLASSES), "คลาสที่กำหนดใน CLASSES ต้องมีครบใน dataset"
    for split in SPLITS:
        assert (DATA_DIR / split).is_dir(), f"missing split: {split}"
        assert list_classes(DATA_DIR / split) == reference

# --- เพิ่มส่วนทดสอบไฟล์ผลลัพธ์ .npz จากขั้นตอน Preprocessing ---
from common import PROCESSED_DIR

def test_processed_files_exist():
    """ตรวจสอบว่าสคริปต์ Preprocessing สร้างไฟล์ .npz ครบทุก split หรือไม่"""
    for split in SPLITS:
        filepath = PROCESSED_DIR / f"{split}.npz"
        assert filepath.exists(), f"ไม่พบไฟล์: {filepath}"

def test_processed_data_shapes():
    """ตรวจสอบว่าจำนวนข้อมูล X (features) และ y (labels) ตรงกัน และมิติถูกต้อง"""
    for split in SPLITS:
        data = np.load(PROCESSED_DIR / f"{split}.npz")
        X = data["X"]
        y = data["y"]
        assert len(X) == len(y), f"จำนวน X และ y ไม่เท่ากันในชุด {split}"
        assert X.ndim == 2, f"มิติของ X ต้องเป็น 2D matrix ในชุด {split}"