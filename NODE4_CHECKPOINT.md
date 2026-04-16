# NODE 4: HYPERPARAMETER CONFIG - CHECKPOINT

**Status**: `DONE` (Implemented & Verified)  
**Last Updated**: 2026-04-14

## 🎯 Purpose
Node 4 ทำหน้าที่สร้างไฟล์ Configuration (YAML) สำหรับการ Fine-tuning และ Hyperparameter Tuning ผ่าน Oumi Framework โดยเน้นความง่ายในการ "พกพา" (Portability) เพื่อนำไปรันบนระบบ Cloud หรือ GPU Cluster เช่น Vast.ai

## 🛠 Architecture & Capabilities

Node 4 ถูกออกแบบด้วยโครงสร้าง Modular เช่นเดียวกับ Node 3:
- **BaseConfigGenerator**: จัดการโครงสร้าง YAML มาตรฐานของ Oumi และการจัดการ Path แบบพกพา
- **ClassificationConfigGenerator**: กำหนดค่า Metric พิเศษ (Macro F1) และ Search Space เริ่มต้นสำหรับงานจำแนกเอกสาร
- **ConfigFactory**: ทำหน้าที่เลือก Generator ที่ถูกต้องตามประเภทงาน

### Key Features:
1.  **Portable Path Management**: ใช้ระบบ Path ที่สัมพันธ์กับโปรเจกต์ (Project Root Relative) ทำให้การย้ายโฟลเดอร์ `runs/` ไปรันบนเครื่องอื่น (Vast.ai) ทำได้ทันที
2.  **Resource Optimization**:
    - **QLoRA (4-bit)**: เปิดใช้งานเป็นค่าเริ่มต้นเพื่อรองรับ GPU ทุกระดับ
    - **Disk Saver**: ตั้งค่า `save_total_limit: 1` เพื่อเก็บเฉพาะโมเดลที่ดีที่สุด ป้องกันปัญหาพื้นที่เต็มบน Vast.ai
3.  **Auto-tuning Intelligence**:
    - **Optuna Integration**: กำหนด Search Space สำหรับ Learning Rate, LoRA Params, และ Batch Size ไว้ใน `tune.yaml`
    - **Metric Target**: ตั้งค่าให้ Optuna มองหาค่าสูงสุดของ `eval_f1_macro` เพื่อให้ได้โมเดลที่มีความแม่นยำและสมดุลที่สุด
4.  **Safe-guarding**: เพิ่มระบบ **Early Stopping** เพื่อหยุดการเทรนหากค่าความแม่นยำไม่ดีขึ้น ช่วยประหยัดเวลาและค่าใช้จ่าย GPU

## 📂 File Structure
```text
src/slm_auto_config/
├── node4/
│   ├── base.py           # Core YAML generation logic
│   ├── classification.py # Classification-specific metrics & search space
│   ├── factory.py        # Task router for generator selection
│   ├── models.py         # Pydantic models for hyperparameters & tuning ranges
│   └── __init__.py
```

## 📝 Generated Artifacts (per run)
- `runs/{run_id}/configs/train.yaml`: สำหรับการเทรนแบบ Manual ด้วยค่าคงที่
- `runs/{run_id}/configs/tune.yaml`: สำหรับการทำ Auto Hyperparameter Tuning (Optuna)

## 🚀 Execution Guide (Next Node)
เมื่อได้ไฟล์ Config จาก Node 4 แล้ว สามารถนำไปรันที่เครื่องปลายทาง (Vast.ai) ได้ด้วยคำสั่ง:
```bash
# เทรนปกติ
oumi train -c runs/{run_id}/configs/train.yaml

# ค้นหาค่า Hyperparameter ที่ดีที่สุด
oumi tune -c runs/{run_id}/configs/tune.yaml
```

**Next Phase**: Node 5 - การจัดการกระบวนการ Fine-tuning และการดึงผลลัพธ์ Metrics กลับมาวิเคราะห์
