# SLM Auto-Config Pipeline 🚀

ระบบอัตโนมัติสำหรับการทำ Fine-tuning Small Language Models (SLMs) สำหรับงานภาษาไทยแบบครบวงจร (End-to-End) ตั้งแต่การนำเข้าข้อมูลดิบไปจนถึงการส่งออกโมเดลที่พร้อมใช้งาน

## 🎯 จุดเด่นของโปรเจกต์
- **Full Automation**: เชื่อมต่อ Node 1 (Intake), Node 2 (SDG) และ Node 3 (Formatting) เข้าด้วยกันเพื่อสร้าง Dataset พร้อมเทรนในคำสั่งเดียว
- **Modern Architecture**: ใช้ **Distilabel** สำหรับสร้างข้อมูลสังเคราะห์และ **Oumi** สำหรับการเทรนที่รวดเร็วและมีประสิทธิภาพ
- **Smart Validation**: มีระบบตรวจเช็คและทำความสะอาดข้อมูล (Data Cleaning) ก่อนเข้าสู่กระบวนการเทรน
- **Vast.ai Integrated**: ออกแบบมาเพื่อทำงานร่วมกับ GPU Cluster บนระบบ Cloud ได้ทันที

## 🛠️ โครงสร้างระบบ (Pipeline Nodes)
- `Node 1-3`: **Data Front-end** (Intake, Synthetic Generation, Oumi Formatting)
- `Node 4-5`: **Training Engine** (Hyperparameter Tuning & SLM SFT)
- `Node 6-7`: **Deployment** (Gradio Playground & Multi-format Export)
