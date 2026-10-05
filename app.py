import streamlit as st
import os
import glob
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Chiang Mai Travel RAG Guide", page_icon="🏔️", layout="wide")

st.title("🏔️ Chiang Mai Travel & Gourmet RAG Assistant")
st.caption("ระบบแชตบอตผู้ช่วยท่องเที่ยวและร้านอาหารเด็ดจังหวัดเชียงใหม่ ตอบคำถามแม่นยำด้วยเทคนิค RAG")

@st.cache_resource
def init_rag_pipeline():
    file_paths = sorted(glob.glob("data/*.txt"))
    chunks = []
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            sections = text.split("\n\n")
            for sec in sections:
                if len(sec.strip()) > 20:
                    lines = sec.strip().split("\n")
                    title = lines[0].replace("#", "").strip() if lines[0].startswith("#") else os.path.basename(path)
                    chunks.append({"title": title, "text": sec.strip(), "source": os.path.basename(path)})

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return chunks, model, index

chunks, embed_model, faiss_index = init_rag_pipeline()

def retrieve_context(query, top_k=3):
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = faiss_index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "title": chunks[idx]["title"],
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "score": float(score)
        })
    return results

groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    st.error("⚠️ ไม่พบ GROQ_API_KEY กรุณาตั้งค่าใน Streamlit Community Cloud Secrets")
    st.stop()

groq_client = Groq(api_key=groq_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ! ผมคือผู้ช่วยท่องเที่ยวเชียงใหม่ มีคำถามเกี่ยวกับสถานที่ท่องเที่ยว ร้านอาหาร ที่พัก หรือเทศกาล ถามมาได้เลยครับ 😊"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่... (เช่น ข้าวซอยแม่สายเปิดกี่โมง)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved = retrieve_context(prompt, top_k=3)
    context_text = "\n\n".join([f"[แหล่งที่มา: {c['source']} - {c['title']}]\n{c['text']}" for c in retrieved])

    system_prompt = f"""คุณคือผู้ช่วยตอบคำถามการท่องเที่ยวและร้านอาหารในจังหวัดเชียงใหม่ที่สุภาพและรอบรู้

กฎการตอบคำถาม:
1. ใช้เฉพาะข้อมูลจาก "บริบท" ด้านล่างในการตอบคำถามเท่านั้น ห้ามใช้ความรู้อื่นนอกเหนือจากนี้
2. หากข้อมูลในบริบทไม่เพียงพอที่จะตอบคำถาม ให้ตอบว่า "ไม่พบข้อมูลนี้ในเอกสารคู่มือท่องเที่ยวเชียงใหม่"
3. ทุกครั้งที่ตอบ ให้ระบุแหล่งที่มา (ชื่อไฟล์และหัวข้อ) ที่ใช้ตอบไว้ท้ายคำตอบเสมอ

บริบท:
{context_text}

คำถาม: {prompt}
คำตอบ:"""

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและประมวลคำตอบ..."):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": system_prompt}],
                    temperature=0.2
                )
                answer = response.choices[0].message.content
                st.markdown(answer)

                with st.expander("📚 ดูเอกสารอ้างอิงที่ค้นพบ (Retrieved Context)"):
                    for idx, c in enumerate(retrieved, 1):
                        st.write(f"**{idx}. {c['title']}** (ไฟล์: `{c['source']}`, Similarity Score: {c['score']:.4f})")
                        st.caption(c['text'])

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเรียก LLM: {e}")