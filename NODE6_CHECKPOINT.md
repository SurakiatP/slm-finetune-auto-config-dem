# NODE 6: INFERENCE PLAYGROUND - CHECKPOINT

**Status**: `DONE` (Production Ready)  
**Last Updated**: 2026-04-16

## 🎯 Purpose
Node 6 คือพื้นที่ทดสอบ (Playground) แบบ Interactive เพื่อให้ผู้ใช้สามารถทดลองนำโมเดลที่ Fine-tuned เสร็จแล้วมาใช้งานจริงผ่านหน้าจอ GUI (Gradio) โดยไม่ต้องเขียนโค้ดเพิ่ม เพื่อตรวจสอบพฤติกรรมของโมเดลก่อนนำไป Deploy จริง

## 🛠 Architecture & Capabilities

Node 6 ถูกออกแบบให้มีความยืดหยุ่นสูงและฉลาดในการค้นหาทรัพยากร:
- **Model Discovery Layer**: ระบบค้นหารันล่าสุดอัตโนมัติ โดยจะไล่ลำดับจาก `final_output` -> `trial_best` -> `trial_*` เพื่อให้มั่นใจว่าผู้ใช้จะได้ทดสอบโมเดลที่ดีที่สุดเสมอ
- **ClassificationInferencer**: Wrapper สำหรับการทำ Inference โดยเฉพาะ รองรับการโหลด LoRA Adapter ลงบน Base Model อย่างรวดเร็ว
- **ResponseParser**: ระบบดัดแปลงผลลัพธ์ (Post-processing) ที่สามารถดึง JSON ออกจากข้อความดิบของโมเดลได้อย่างแม่นยำ

### Key Features:
1.  **Dynamic Fuzzy Label Mapping**: 
    - **Highlight**: เลิกใช้การ Hardcode ชื่อคลาส! ระบบจะอ่านรายชื่อ Labels ที่ถูกต้องจาก `data_report.json` โดยตรง
    - **Snapped Logic**: หากโมเดลตอบเพี้ยน (เช่น ตอบคำว่า "NDA" แทนที่จะตอบ "ข้อตกลงรักษาความลับ (NDA)") ระบบจะทำการเปรียบเทียบข้อความ (Fuzzy Match) และดึงค่าที่ถูกต้องที่สุดกลับมาให้โดยอัตโนมัติ
2.  **Standard Chat Template Compliance**: 
    - รองรับการถาม-ตอบแบบแยกบทบาท (System/User) ตามมาตรฐานสากล ทำให้โมเดล Instruct ทำงานได้อย่างเต็มประสิทธิภาพและลดการตอบว่า `unknown` โดยใช่เหตุ
3.  **Real-time Confidence Score**: 
    - แสดงผลลัพธ์พร้อมค่า "ความมั่นใจ" (Confidence Score) เพื่อให้ผู้ใช้ตัดสินใจได้ว่าควรเชื่อถือคำตอบนั้นหรือไม่
4.  **Lightweight & Fast**: 
    - ปรับแต่งมาเพื่อ **Qwen2.5-0.5B** ทำให้การตอบสนองแทบจะทันที (Near-instant) แม้รันบน CPU หรือ GPU ขนาดเล็ก

## 📂 File Structure
```text
src/slm_auto_config/
├── node6/
│   ├── base.py           # Abstract Base สำหรับระบบ Inferencer
│   ├── classification.py # Logic หลักของงานจำแนก (Inference + Dynamic Labels)
│   ├── parser.py         # Regex-based JSON extractor สำหรับ LLM output
│   ├── playground.py     # Gradio UI definitions และ Event handlers
│   └── __init__.py
├── launch_playground.py  # Entry point สำหรับรันหน้าจอทดสอบ
```

## 🚀 Execution Guide
สามารถรัน Playground ได้ทันทีจาก Root Project:
```bash
python launch_playground.py --run_id test_full_pipeline_001
```
*(หากไม่ใส่ `--base_model` ระบบจะใช้ Qwen2.5-0.5B-Instruct เป็นค่าเริ่มต้น)*

## 📝 Document/Data Contract
- **Input**: ข้อความ (Plain Text)
- **Output**: JSON Predict + Label ที่ถูก Normalize แล้ว + Confidence Score

**Next Phase**: Node 7 - การประเมินผลเชิงลึกและการส่งออกโมเดล (Export)
