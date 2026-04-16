# NODE 5: FINE-TUNING & EVALUATION - CHECKPOINT

**Status**: `DONE` (Implemented & Environment Ready)  
**Last Updated**: 2026-04-14

## 🎯 Purpose
Node 5 คือสะพานเชื่อมระหว่างการเตรียมตั้งค่า (Node 4) และการนำโมเดลไปรันจริงบน GPU (Vast.ai) โดยรับผิดชอบตั้งแต่การติดตั้งสภาพแวดล้อม การสั่งรันแบบอัตโนมัติ ไปจนถึงการวิเคราะห์ผลลัพธ์และสรุปรายงานเป็นรูปเล่ม

## 🛠 Architecture & Capabilities

Node 5 ถูกออกแบบให้รองรับการทำงานแบบ Remote-First:
- **ExecutorGenerator**: สร้างสคริปต์สั่งการ (.sh) ที่พกพาไปรันบน Vast.ai ได้ทันที
- **MetricsAnalyser**: ตัวสแกนหา "โมเดลที่ดีที่สุด" จากหลายๆ Trial และสรุปค่า Metrics ให้เป็นมาตรฐาน
- **Visualizer**: ระบบวาดกราฟสถิติ, Confusion Matrix และสร้าง PDF Training Report

### Key Features:
1.  **Vast.ai Optimization**:
    - `setup_vast.sh`: ติดตั้ง Oumi และ Dependencies ทั้งหมดบนคลาวด์ในคลิกเดียว
    - `oumi[gpu]`: ติดตั้งเวอร์ชันรองรับ GPU เป็นค่าเริ่มต้น
2.  **Smart Sync with rsync**:
    - `sync_to_vast.sh` / `sync_from_vast.sh`: ใช้เทคโนโลยี `rsync` เพื่อส่งข้อมูลก้อนใหญ่ (Model Weights) ได้อย่างรวดเร็วและสามารถส่งต่อจากจุดที่ค้างได้
3.  **Cyclic Loop Support (Auto Mode)**:
    - รองรับการคัดเลือก **Best Trial** จากการทำ Optuna Tuning อัตโนมัติ
    - จดจำตัวเลือกโมเดลที่ดีที่สุดเพื่อส่งไม้ต่อให้ Node 6 (Inference)
4.  **Executive Reporting**:
    - **PDF Report**: สรุปผลการเทรนอย่างละเอียดพร้อมกราฟเปรียบเทียบ
    - **Separate Testing**: รัน `oumi evaluate` บนชุด Test Set (10%) ที่แยกไว้ เพื่อยืนยันความสามารถของโมเดลกับข้อมูลจริงภายนอก

## 📂 File Structure
```text
src/slm_auto_config/
├── node5/
│   ├── executor.py       # Bash script automation for Oumi & Sync
│   ├── analyser.py       # Trials results parsing & Best model selection
│   ├── visualizer.py     # Confusion Matrix plotting & PDF Reporting
│   ├── models.py         # Evaluation & Analysis schemas
│   └── __init__.py
runs/
└── setup_vast.sh         # Global installation script for remote server
```

## 📝 Generated Artifacts (per run)
- `runs/{run_id}/scripts/run_train.sh` & `run_tune.sh`
- `runs/{run_id}/scripts/run_eval.sh`
- `runs/{run_id}/evaluation/metrics.json`
- `runs/{run_id}/evaluation/report.pdf` (PDF สรุปผล)

## 🚀 Execution Guide
1.  **Setup**: รัน `bash runs/setup_vast.sh` บน Vast.ai
2.  **Upload**: รัน `./sync_to_vast.sh` จากเครื่อง Local
3.  **Train**: รัน `./run_tune.sh` หรือ `./run_train.sh` บน Vast.ai
4.  **Evaluate**: รัน `./run_eval.sh` บน Vast.ai เพื่อวัดผลครั้งสุดท้าย
5.  **Sync Back**: รัน `./sync_from_vast.sh` จากเครื่อง Local เพื่อดึงผลสรุปและ PDF กลับมา
