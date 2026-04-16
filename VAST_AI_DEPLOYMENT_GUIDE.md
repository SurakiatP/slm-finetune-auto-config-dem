# 🚀 Vast.ai Deployment Guide: SLM Auto Config (Latest Version)

คู่มือฉบับปรับปรุงล่าสุด สำหรับการนำ Pipeline ไปรันบน Vast.ai อย่างราบรื่น

---

## 1. การเตรียมเครื่อง (Instance Creation)

1.  **Hardware:** แนะนำ GPU VRAM 24GB+ (RTX 3090, 4090, A10)
2.  **OS/Docker:** เลือก `pytorch/pytorch` ล่าสุด
3.  **Storage:** **50GB - 100GB** (สำคัญมาก)

---

## 2. การเตรียม Code บน Vast.ai (GitHub Method)

เมื่อ SSH เข้าเครื่อง Vast.ai ได้ครั้งแรก ให้รันกลุ่มคำสั่งนี้เพื่อสร้าง Folder และโหลดโค้ด:

```bash
# 1. สร้าง Folder หลัก
mkdir -p ~/slm-auto-config

# 2. Clone Code ลงไป (หรือใช้ SCP อัปโหลดโค้ดทั้งหมดขึ้นมา)
git clone [URL_REPRO_ของคุณ] ~/slm-auto-config

# 3. เข้าไปที่โฟลเดอร์โปรเจกต์
cd ~/slm-auto-config

# 4. รันสคริปต์ Setup สภาพแวดล้อม
bash setup_vast.sh
```

---

## 3. การเตรียมข้อมูล (จากเครื่อง Local ของคุณ)

### 3.1 ส่งไฟล์ Dataset (SCP)
หากไฟล์ `synthetic_data.json` อยู่ที่เครื่องคอมคุณ และถูก Ignore จาก Git ให้ส่งผ่านคำสั่งนี้:
```bash
# รันที่เครื่องคอมพิวเตอร์ของคุณ
scp -P [VAST_PORT] synthetic_data.json root@[VAST_IP]:~/slm-auto-config/
```

### 3.2 ส่ง Config และโครงสร้างสคริปต์ (RSYNC)
```bash
# รันที่เครื่องคอมพิวเตอร์ของคุณ
bash runs/[YOUR_RUN_ID]/scripts/sync_to_vast.sh root@[VAST_IP] [VAST_PORT]
```

---

## 4. เริ่มการ Training / Tuning (บน Vast.ai)

ที่ Terminal ของ Vast.ai (`cd ~/slm-auto-config`):

### 🎯 เริ่มการจูนอัตโนมัติ (Search Best Hyperparameters)
```bash
bash runs/[YOUR_RUN_ID]/scripts/run_tune.sh
```

### 🎯 เริ่มการเทรนแบบระบุค่าคงที่
```bash
bash runs/[YOUR_RUN_ID]/scripts/run_train.sh
```

---

## 5. การทดสอบและดูผล (Interactive)

### 🧪 ลองเล่นโมเดล (Playground)
```bash
python launch_playground.py --run_id [YOUR_RUN_ID] --share
```
*ระบบจะสร้างลิงก์ **https://xxxx.gradio.live** ให้คุณเปิดทดสอบจากเครื่องที่ไหนก็ได้ในโลก!*

### 📊 ดูรายงานผลการเทรน
หลังเทรนเสร็จ สามารถดูสรุปผลในรูป PDF และ Confusion Matrix ได้ที่โฟลเดอร์:
`runs/[YOUR_RUN_ID]/evaluation/reports/`

---

## 6. การ Export และดาวน์โหลดกลับ

### 📦 รวมร่างโมเดล (บน Vast.ai)
ทำการ Merge Weights และแปลงเป็น GGUF/ONNX:
```bash
python run_export.py --run_id [YOUR_RUN_ID] --formats safetensors gguf onnx
```

### 📥 ดาวน์โหลดกลับเครื่อง (บน Local Machine)
```bash
bash runs/[YOUR_RUN_ID]/scripts/sync_from_vast.sh root@[VAST_IP] [VAST_PORT]
```

---

## ⚠️ ข้อควรระวัง (Tips)
-   **SSH Terminal:** หากเน็ตหลุดระหว่างเทรน แนะนำให้ใช้ `screen` หรือ `tmux` บน Vast.ai เพื่อให้โปรเซสไม่ค้าง
-   **Storage Management:** ลบโฟลเดอร์ `runs/xxx/training/output` ที่ไม่ใช้ออกบ้างเพื่อประหยัดที่ (แต่เก็บโฟลเดอร์ `export/` ไว้)

---
**สำเร็จ! คุณมีระบบ SLM Fine-Tuning ที่สมบูรณ์แบบพร้อมใช้งานแล้วครับ** ⚖️🤖
