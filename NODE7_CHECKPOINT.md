# NODE 7: DETAILED EVALUATION & EXPORT - CHECKPOINT

**Status**: `DONE` (Production Ready)  
**Last Updated**: 2016-04-16

## 🎯 Purpose
Node 7 รับหน้าที่เป็นขั้นตอนสุดท้ายของ Pipeline โดยทำหน้าที่วัดผลประสิทธิภาพ (Evaluation) ขั้นสูงเฉพาะทางสำหรับงานจำแนกเอกสาร (Classification) พร้อมทั้งออกรายงานสรุปผลรูปเล่ม (PDF) และเตรียม Model Artifacts สำหรับการนำไปใช้งานในระบบอื่น (Export)

## 🛠 Architecture & Capabilities

Node 7 เน้นความเป็นระเบียบและการรายงานผลที่ผู้บริหารหรือนักวิเคราะห์นำไปใช้ต่อได้ทันที:
- **Rich Evaluator**: เลิกใช้แค่ค่า Loss ในการวัดผล แต่ใช้ `scikit-learn` เพื่อคำนวณ Precision, Recall, F1-Score สำหรับทุกๆ คลาส (Label-wise Metrics)
- **Visualizer Professional**: ระบบกราฟิกที่เปลี่ยนตัวเลขใน JSON ให้เป็น Confusion Matrix ที่ดูง่าย และบันทึกรวมเล่มเป็น PDF Report
- **Export Manager**: ผสมผสานโมเดล Base + Adapter และสร้าง Model Card มาตรฐาน เพื่อให้พร้อมต่อยอดใน HuggingFace หรือระบบภายในอื่นๆ

### Key Features:
1.  **Dynamic Label Integration**: 
    - ดึงข้อมูล Label จาก `data_report.json` ทำให้การคำนวณ Metrics เป็นอัตโนมัติตามข้อมูลจริงของผู้ใช้ (User Data Dependent)
2.  **High-Fidelity Classification Reporting**: 
    - คำนวณค่า **Macro F1-Score** และแสดงตารางคลาสที่โมเดลอาจจะยังจำแนกได้ไม่ดีพอ (Confusion Matrix) เพื่อให้ผู้ใช้กลับไปเพิ่ม Seed Data ได้ถูกจุด
3.  **PDF Automated Reporting**: 
    - สร้างไฟล์ `report.pdf` ที่รวมทั้งสถิติการเทรน, กราฟ Loss, และผลการประเมินชุด Test Set ไว้ในเล่มเดียว
4.  **Model Card Generation**: 
    - สร้างไฟล์ `model_card.md` อัตโนมัติ โดยระบุรายละเอียด Model Architecture, Hyperparameters ที่ใช้ และประสิทธิภาพที่ได้ เพื่อความโปร่งใส (Model Governance)

## 📂 File Structure
```text
src/slm_auto_config/
├── node7/
│   ├── run_eval.py       # คลาสหลักสำหรับรันประเมินผลและคำนวณ Metrics รายคลาส
│   ├── generator.py      # ระบบสร้าง Model Card และ Markdown Documentation
│   └── __init__.py
├── run_export.py         # สคริปต์สำหรับส่งออก Adapter และ Merged Weights
```

## 🚀 Execution Guide
เมื่อทำการ `sync_from_vast.sh` นำผลลัพธ์กลับมาที่เครื่อง Local แล้ว สามารถรันชุดสรุปผลได้คือ:
```bash
# รันการประเมินผลละเอียดและออก PDF Report
python src/slm_auto_config/node7/run_eval.py [RUN_ID]

# ส่งออกโมเดลและ Model Card
python run_export.py --run_id [RUN_ID]
```

## 📝 Document/Data Contract
- **Input**: `predictions.jsonl` (จาก Oumi Evaluate) + `data_report.json`
- **Output**: 
  - `evaluation/metrics.json`
  - `evaluation/report.pdf`
  - `export/adapter/`
  - `export/model_card.md`
