"""ทดสอบว่าโมเดลเทรนและทำนายบนฟีเจอร์จาก extract_features ได้จริง (ใช้ภาพสังเคราะห์ ไม่ต้องมี dataset)"""

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import extract_features


def make_leaf(color, seed):
    """ภาพสังเคราะห์: พื้นสีหลัก + noise เล็กน้อย"""
    rng = np.random.default_rng(seed)
    arr = np.clip(np.array(color) + rng.normal(0, 15, (64, 64, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_dataset(n_per_class=15):
    colors = {"Healthy": (40, 160, 40), "Late_blight": (90, 60, 30), "Leaf_Mold": (180, 170, 60)}
    X, y = [], []
    for i, (label, color) in enumerate(colors.items()):
        for k in range(n_per_class):
            X.append(extract_features(make_leaf(color, seed=i * 100 + k)))
            y.append(label)
    return np.stack(X), np.array(y)


def test_model_learns_simple_classes():
    X, y = build_dataset()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    assert (model.predict(X) == y).mean() >= 0.95


def test_prediction_returns_class_names():
    X, y = build_dataset(n_per_class=5)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    pred = model.predict(X[:3])
    assert set(pred) <= set(y)
    proba = model.predict_proba(X[:3])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
