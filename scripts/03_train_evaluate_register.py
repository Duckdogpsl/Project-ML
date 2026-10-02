"""ขั้นที่ 3: เทรนหลายโมเดล เลือกตัวที่ดีที่สุดจาก val ประเมินบน test แล้ว register ลง MLflow

- ทุกโมเดลที่ลองถูกบันทึกเป็น child run (Experiment Tracking)
- โมเดลที่ดีที่สุดถูก register ชื่อ tomato-leaf-classifier และตั้ง alias 'champion' (Model Registry)
- ถ้า val accuracy ต่ำกว่า MIN_VAL_ACCURACY จะไม่ register และจบด้วย exit code != 0 (quality gate)

ค่าเริ่มต้นลอง random_forest และ svc_rbf (logreg ช้ามากกับข้อมูลเต็ม แต่เร็วกับ sample)
เลือกเองได้ด้วย env เช่น  MODELS=logreg,svc_rbf
"""

import inspect
import os
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (
    MIN_VAL_ACCURACY,
    MODEL_ALIAS,
    MODEL_NAME,
    PROCESSED_DIR,
    SEED,
    setup_mlflow,
)

DEFAULT_MODELS = "random_forest,svc_rbf"


def build_candidates():
    return {
        "logreg": (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
            {"model": "LogisticRegression", "C": 0.05},
        ),
        "random_forest": (
            make_pipeline(
                RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
            ),
            {"model": "RandomForest", "n_estimators": 200},
        ),
        "svc_rbf": (
            make_pipeline(
                StandardScaler(),
                PCA(n_components=150, random_state=SEED),
                SVC(C=10, gamma="scale", random_state=SEED),
            ),
            {"model": "PCA+SVC(rbf)", "pca_components": 150, "C": 10},
        ),
    }


def load(split):
    d = np.load(PROCESSED_DIR / f"{split}.npz")
    return d["X"], d["y"]


def evaluate(model, X, y, prefix):
    pred = model.predict(X)
    return pred, {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_f1_macro": f1_score(y, pred, average="macro"),
    }


def train():
    setup_mlflow()
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    candidates = build_candidates()
    selected = [m.strip() for m in os.getenv("MODELS", DEFAULT_MODELS).split(",")]

    with mlflow.start_run(run_name="train_evaluate_register") as parent:
        mlflow.set_tag("ml.step", "train_evaluate_register")
        results = {}

        # ---------- เทรนและเทียบทุกโมเดลบน val ----------
        for name in selected:
            model, params = candidates[name]
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.log_params(params)
                t0 = time.time()
                model.fit(X_train, y_train)
                _, metrics = evaluate(model, X_val, y_val, "val")
                metrics["train_seconds"] = round(time.time() - t0, 1)
                mlflow.log_metrics(metrics)
                results[name] = (model, metrics, child.info.run_id)
                print(f"{name:>14}: val_acc={metrics['val_accuracy']:.4f} "
                      f"f1={metrics['val_f1_macro']:.4f} ({metrics['train_seconds']}s)")

        best_name = max(results, key=lambda n: results[n][1]["val_accuracy"])
        best_model, best_val, _ = results[best_name]
        print(f"\nBest model: {best_name}")

        # ---------- ประเมินตัวที่ดีที่สุดบน test ----------
        test_pred, test_metrics = evaluate(best_model, X_test, y_test, "test")
        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({**best_val, **test_metrics})
        report = classification_report(y_test, test_pred, digits=4)
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_pred, ax=ax, xticks_rotation=60, colorbar=False
        )
        ax.set_title(f"Confusion matrix (test) — {best_name}")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # ---------- quality gate ----------
        if best_val["val_accuracy"] < MIN_VAL_ACCURACY:
            mlflow.set_tag("registered", "false")
            raise SystemExit(
                f"val_accuracy {best_val['val_accuracy']:.4f} < {MIN_VAL_ACCURACY} — ไม่ register โมเดล"
            )

        # ---------- Model Registry ----------
        # MLflow รุ่นใหม่บันทึกโมเดลด้วย skops ซึ่งต้องระบุชนิด object ที่ไว้ใจ (เช่น Tree ของ RandomForest)
        # เราสร้างโมเดลเองจึงไว้ใจได้ — รุ่นเก่าที่ไม่มีพารามิเตอร์นี้จะข้ามไป
        extra = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        info = mlflow.sklearn.log_model(
            best_model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_train[:5], best_model.predict(X_train[:5])),
            input_example=X_train[:2],
            **extra,
        )
        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
        client.set_model_version_tag(MODEL_NAME, version, "algorithm", best_name)
        client.set_model_version_tag(
            MODEL_NAME, version, "test_accuracy", f"{test_metrics['test_accuracy']:.4f}"
        )
        mlflow.set_tag("registered", "true")
        print(f"Registered {MODEL_NAME} v{version} as @{MODEL_ALIAS} "
              f"(test_acc={test_metrics['test_accuracy']:.4f}, run={parent.info.run_id})")


if __name__ == "__main__":
    train()
