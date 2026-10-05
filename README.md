# NLP-SubTest2-1
SubTest2

# 🏔️️ Chiang Mai Travel & Gourmet RAG Assistant

แชตบอตผู้ช่วยอัจฉริยะสำหรับการท่องเที่ยว ร้านอาหาร และที่พักในจังหวัดเชียงใหม่ พัฒนาด้วยเทคนิค **Retrieval-Augmented Generation (RAG)** ตอบคำถามอ้างอิงจากเอกสารข้อมูลจริงด้วยความแม่นยำสูง

---

## 🌟 จุดเด่นของระบบ (Features)

* **RAG Pipeline แม่นยำสูง:** ค้นหาและดึงข้อมูลอ้างอิงเฉพาะจากเอกสารคู่มือการท่องเที่ยวเชียงใหม่ ไม่มั่วคำตอบ (No Hallucination)
* **Streaming Response:** ตอบคำถามแบบพิมพ์ทีละตัวอักษรเรียลไทม์ผ่าน Groq API เพิ่มประสบการณ์การใช้งานที่ลื่นไหล
* **การค้นหาภาษาไทย:** รองรับ Vector Embedding ภาษาไทยและหลายภาษาด้วยโมเดล `paraphrase-multilingual-MiniLM-L12-v2`
* **ระบบจัดเก็บดรรชนีความเร็วสูง:** ค้นหาข้อความด้วย **FAISS (IndexFlatIP)** คำนวณ Cosine Similarity ได้อย่างรวดเร็ว
* **แสดงเอกสารอ้างอิง (Retrieved Context):** มีเมนูให้ผู้ใช้สามารถเปิดดูที่มาของข้อมูล ค่าน้ำหนักความคล้ายคลึง (Similarity Score) และเนื้อหาต้นฉบับได้
* **UI/UX สะดวกใช้งาน:** สร้างด้วย Streamlit พร้อมปุ่มคำถามตัวอย่าง และฟังก์ชันล้างประวัติการสนทนา

---

## 🛠️ สถาปัตยกรรมระบบ (Architecture & Tech Stack)

1. **User Interface:** [Streamlit](https://streamlit.io/)
2. **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
3. **Vector Database / Indexing:** [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`)
4. **LLM Engine:** Groq API (`openai/gpt-oss-20b` หรือ `llama-3.1-8b-instant`)
5. **Data Source:** ไฟล์ข้อความ (`data/*.txt`) เก็บรายละเอียดสถานที่ท่องเที่ยวและร้านอาหาร

---

## ⚙️ การทำงานของระบบ RAG (How it Works)

1. **Document Chunking:** ระบบอ่านไฟล์ในโฟลเดอร์ `data/*.txt` และแยกเนื้อหาออกเป็นชิ้นๆ (Chunks) ตามหัวข้อและความยาว
2. **Vector Indexing:** แปลงข้อความข้อความแต่ละ Chunk เป็น Vector Embedding แล้วบันทึกลงใน FAISS Index
3. **Context Retrieval:** เมื่อผู้ใช้ถามคำถาม ระบบจะแปลงคำถามเป็น Vector แล้วใช้ FAISS ดึงข้อมูลที่มีค่าความคล้ายคลึงสูงสุด **Top-5** (`top_k=5`)
4. **Prompt Engineering & Generation:** นำข้อความบริบทที่ดึงได้ (Context) มารวมกับคำถาม ส่งให้ Groq LLM ประมวลผลและส่งคำตอบแบบ Stream กลับมาแสดงผล

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
.
├── app.py                   # โค้ดหลักของแอปพลิเคชัน Streamlit
├── requirements.txt         # รายชื่อคลังไลบรารีที่จำเป็น
├── README.md                # เอกสารอธิบายโปรเจกต์
└── data/                    # โฟลเดอร์เก็บเอกสารคู่มือท่องเที่ยว (ไฟล์ .txt)
    ├── 01_สถานที่ท่องเที่ยว.txt
    ├── 04_ร้านอาหารท้องถิ่นและมิชลินไกด์.txt
    └── ...