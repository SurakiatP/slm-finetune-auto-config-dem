# SLM Auto-Config Pipeline 🚀

ระบบจัดการ Pipeline สำหรับการทำ Fine-tuning Small Language Models (SLMs) ตั้งแต่การนำเข้าข้อมูล (Data Intake) จนถึงชุดเตรียมข้อมูลสังเคราะห์ (SDG) และ Oumi-formatted datasets แบบอัตโนมัติ

## 🌟 Key Features
- **Node 1 (Intake)**: รับไฟล์ CSV, JSON, JSONL พร้อมระบบ Auto-mapping และ Quarantine ข้อมูลเสีย
- **Node 2 (SDG)**: สร้างข้อมูลสังเคราะห์คุณภาพสูงด้วย Meta-Prompting และ Semantic Deduplication (Distilabel + FAISS)
- **Node 3 (Split)**: แบ่งชุดข้อมูล Train/Val/Test พร้อมแปลงเป็นฟอร์แมต Oumi Chat Template ทันที
- **One-Command Orchestration**: รันทุกขั้นตอน Node 1-3 ผ่านทาง CLI ในคำสั่งเดียว

## 🚀 Quick Start

### 1. Setup Environment
ติดตั้ง Dependencies และเตรียมไฟล์ API Key ใน `.env`:
```powershell
pip install -r requirements.txt
cp .env.example .env  # กรอก OPENROUTER_API_KEY
```

### 2. Run the Full Pipeline
เริ่มสร้าง Dataset สำหรับเทรนโหมด Classification ได้จากเครื่องคุณ:
```powershell
python run_full_pipeline.py --task "คำอธิบายงานของคุณ" --input "data/raw/seed.csv" --count 100
```

## 📂 Project Structure
- `src/slm_auto_config/`: Core Logic (แบ่งตาม Node 1-7)
- `tests/`: ชุดทดสอบระบบแยกตาม Node (Unit Tests)
- `runs/`: โฟลเดอร์เก็บผลลัพธ์จากการรัน (Ignore ใน Git)
- `run_full_pipeline.py`: ตัวสั่งการหลัก (Frontend Pipeline)
- `launch_playground.py`: หน้าจอทดลองโมเดลหลังเทรนเสร็จ (Node 6)

## ☁️ Deployment
สำหรับขั้นตอนการเทรนบน GPU (Node 4-5) ให้ศึกษาที่ [VAST_AI_DEPLOYMENT_GUIDE.md](VAST_AI_DEPLOYMENT_GUIDE.md)

---
*Maintained by Park - Advanced SLM Fine-tuning Workflow.*
