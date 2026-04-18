# NODE 3: SPLIT DATA - CHECKPOINT

**Status**: `DONE` (Fully Integrated End-to-End)  
**Last Updated**: 2026-04-18

## 🎯 Purpose
Node 3 ทำหน้าที่รับข้อมูลจาก Node 2 (Synthetic Data) และ Node 1 (Seed Data) เพื่อทำการแบ่งข้อมูลเป็นชุด Train, Validation, และ Test พร้อมทั้งแปลงให้อยู่ในรูปแบบ Oumi `text_sft` (Conversation JSONL) โดยใช้มาตรฐาน **Chat Template** (System/User/Assistant) เพื่อประสิทธิภาพสูงสุดของโมเดล Instruct

## 🛠 Architecture & Capabilities

เรารีแฟกเตอร์ Node 3 ให้เป็นแบบ **Modular Architecture** โดยใช้ Pattern ดังนี้:
- **Strategy Pattern**: แยก Logic ตามประเภทงาน (Task-specific)
- **Factory Pattern**: ใช้ `get_splitter()` เพื่อสร้างออบเจกต์ตาม `task_type` โดยไม่ต้องแก้โค้ดหลัก

### Key Features:
1.  **Dynamic Label Detection**: ระบบสแกนหา Labels จากข้อมูลจริงโดยอัตโนมัติ ทำให้สคริปต์เป็น **Domain-Agnostic** (รองรับทุกโดเมนข้อมูล)
2.  **Stratified Splitting**: ใช้ `scikit-learn` เพื่อรักษาความสมดุลของหมวดหมู่ (Label Balance) ในทุกๆ ชุดข้อมูล (Train/Val/Test)
3.  **Seed Data Priority**: ปฏิบัติตามกฎ `AGENT_RULES.md` โดยใช้ Seed Data ที่เป็น Ground Truth สำหรับชุด Validation และ Test เป็นลำดับแรก
4.  **Standard Chat Template**: เลิกใช้การยุบรวม Prompt (Flattening) และเปลี่ยนมาใช้แยกบทบาท: `system` (Persona), `user` (Instruction + Content), และ `assistant` (Target)
5.  **Multi-format Bridge**: เพิ่มฟังก์ชัน `_load_data_flexibly` ทำให้อ่านไฟล์ได้ทั้ง `.json` และ `.jsonl` เพื่อรับช่วงต่อจาก Node 1 ได้ทันที
6.  **Pipeline Orchestration**: เชื่อมต่อเข้ากับ `run_full_pipeline.py` รันต่อเนื่องจาก Node 1 ➡️ 2 ➡️ 3 ในคำสั่งเดียว

## 📂 File Structure
```text
src/slm_auto_config/
├── node3/
│   ├── base.py           # Abstract Class สำหรับ Splitter ทุกประเภท
│   ├── classification.py # Logic เฉพาะสำหรับงาน Classification (Corrected Template)
│   ├── factory.py        # ตัวเลือกว่าจะใช้ Splitter ตัวไหน (Task Router)
│   ├── models.py         # Pydantic models สำหรับข้อมูล Node 3
│   └── __init__.py
└── utils.py              # Common JSON/JSONL I/O utilities
```

## 📝 Document/Data Contract

### Output JSONL (Oumi Format):
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert document classifier."
    },
    {
      "role": "user",
      "content": "Your task is to identify... Text: [CONTENT]"
    },
    {
      "role": "assistant",
      "content": "{\"label\": \"A\"}"
    }
  ]
}
```

### Data Report (`data_report.json`):
บันทึกสถิติการแบ่งข้อมูลและสัดส่วนของแต่ละ Label ในทุก Split เพื่อใช้ในการตรวจสอบคุณภาพก่อนเทรน

## 🚀 Next Step
- **Node 4**: การสร้าง Oumi Config (YAML) เพื่อรับไฟล์จาก Node 3 ไปทำการ Train/Tune
