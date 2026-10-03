# Pipeline Orchestration (Prefect + Airflow)

ร้อยสคริปต์ทั้ง 4 ขั้นให้รันต่อกันอัตโนมัติ มีลำดับ มี log มี retry และตั้งเวลาได้ โดยไม่ต้องแก้สคริปต์เดิม

```
01_data_validation ──► 02_data_preprocessing ──► 03_train_evaluate_register ──► 04_load_and_predict
   (หยุดถ้าข้อมูลเสีย)        (retry 1 ครั้ง)          (quality gate val_acc ≥ 0.70)        (smoke test @champion)
```

## วางไฟล์

```
project/
├── scripts/      01_... 02_... 03_... 04_... common.py make_sample_dataset.py
├── pipelines/    prefect_flow.py  airflow_dag.py  README.md
├── dataset/      tomato/ (เต็ม)  sample/ (เล็ก)
└── requirements.txt   ← เพิ่ม prefect>=3
```

## Prefect (ตัวหลัก)

```bash
pip install "prefect>=3"

# รันครั้งเดียว
python pipelines/prefect_flow.py                                   # ใช้ dataset/sample
python pipelines/prefect_flow.py --data-dir dataset/tomato         # ใช้ข้อมูลเต็ม

# เปิด UI (อีก terminal) ดูกราฟ task, log, artifact สรุปผล
prefect server start            # http://127.0.0.1:4200

# ตั้งเวลา retrain ทุกวัน 02:00 (เปิด process นี้ค้างไว้ กด Run จาก UI ได้ด้วย)
python pipelines/prefect_flow.py --serve
```

พารามิเตอร์: `--data-dir`, `--classes`, `--max-per-class`, `--models`, `--min-val-accuracy`

## Airflow (ทางเลือก — ต้อง WSL2/Docker)

```bash
pip install apache-airflow
export TOMATO_PROJECT_DIR=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/pipelines
airflow standalone              # http://localhost:8080 → DAG tomato_leaf_mlops_pipeline → Trigger
```

## จุดที่ใช้ตอบในรายงาน/พรีเซนต์

| หัวข้อ | ทำอย่างไร |
|---|---|
| DAG / dependency | task ถัดไปรอ task ก่อนหน้า (`wait_for` / `>>`) ขั้นใดล้ม ขั้นหลังไม่รัน |
| Fail fast | สคริปต์ `raise SystemExit` → exit code ≠ 0 → task failed |
| Quality gate | ขั้น 3 ไม่ register ถ้า val_acc < `MIN_VAL_ACCURACY` |
| Retry | preprocessing retry 1 ครั้ง (I/O ไฟล์ภาพ) |
| Parameterize | เปลี่ยน dataset/โมเดล/threshold ได้ตอนสั่งรัน ไม่ต้องแก้โค้ด |
| Scheduling | cron `0 2 * * *` (retrain รายวัน) |
| Observability | log ทุกบรรทัดใน UI + artifact สรุปผล + ผลทุก run อยู่ใน MLflow |
