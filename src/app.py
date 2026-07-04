import streamlit as st
import chromadb
from pypdf import PdfReader
from openai import OpenAI
import tempfile
import os
import subprocess
import re
from chunker import fixed_chunker, sentence_chunker, deduplicate_chunks

st.set_page_config(page_title="Local RAG - Foundry", page_icon="🤖", layout="wide")
st.title("🤖 Local RAG with Foundry Local")
st.caption("Tamamen lokalde çalışan yapay zeka — internet yok, API ücreti yok")

@st.cache_resource
def get_foundry_port():
    try:
        result = subprocess.run(["foundry", "service", "status"], capture_output=True, text=True)
        match = re.search(r'http://127\.0\.0\.1:(\d+)', result.stdout)
        if match:
            return match.group(1)
    except:
        pass
    return "51250"

@st.cache_resource
def get_client():
    port = get_foundry_port()
    return OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="foundry")

@st.cache_resource
def get_chroma():
    return chromadb.Client()

client = get_client()
chroma = get_chroma()

if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

with st.sidebar:
    st.header("📄 Doküman Yükle")
    
    # Chunking stratejisi seç
    st.subheader("⚙️ Ayarlar")
    chunk_strategy = st.selectbox(
        "Chunking Stratejisi",
        ["Sentence-based (Önerilen)", "Fixed-size"],
        help="Sentence-based cümle ortasında kesmez, daha kaliteli sonuç verir"
    )
    
    uploaded_files = st.file_uploader("PDF seç (birden fazla olabilir)", type="pdf", accept_multiple_files=True)

    if st.button("📥 Yükle ve İşle", disabled=not uploaded_files):
        all_chunks = []
        loaded_names = []

        for uploaded_file in uploaded_files:
            with st.spinner(f"{uploaded_file.name} işleniyor..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                reader = PdfReader(tmp_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() or ""
                os.unlink(tmp_path)

                if not full_text.strip():
                    st.warning(f"⚠️ {uploaded_file.name} okunamadı, atlandı.")
                    continue

                if chunk_strategy == "Sentence-based (Önerilen)":
                    chunks = sentence_chunker(full_text)
                else:
                    chunks = fixed_chunker(full_text)

                all_chunks.extend(chunks)
                loaded_names.append(uploaded_file.name)

        if all_chunks:
            collection_name = "rag_collection"
            try:
                chroma.delete_collection(collection_name)
            except:
                pass
            collection = chroma.create_collection(collection_name)
            all_chunks = deduplicate_chunks(all_chunks)
            collection.add(
                documents=all_chunks,
                ids=[f"chunk_{i}" for i in range(len(all_chunks))]
            )
            st.session_state.collection = collection
            st.session_state.pdf_names = loaded_names
            st.session_state.messages = []
            st.success(f"✅ {len(all_chunks)} parça yüklendi!")

    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**Yüklü Dokümanlar:**")
        for name in st.session_state.pdf_names:
            st.caption(f"📖 {name}")

    st.divider()
    st.markdown("**Model:** Phi-4-mini")
    st.markdown("**Veritabanı:** ChromaDB")
    st.markdown("**Çalışma:** 100% Lokal")

if st.session_state.collection is None:
    st.info("👈 Başlamak için sol panelden bir PDF yükle")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("📎 Kaynaklar"):
                    for i, src in enumerate(msg["sources"]):
                        st.caption(f"Parça {i+1}: {src[:200]}...")

    if soru := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": soru})
        with st.chat_message("user"):
            st.write(soru)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                results = st.session_state.collection.query(query_texts=[soru], n_results=3)
                context = "\n".join(results['documents'][0])
                sources = results['documents'][0]
                response = client.chat.completions.create(
                    model="Phi-4-mini-instruct-cuda-gpu:5",
                    messages=[
                        {"role": "system", "content": "Sen bir akademik makale asistanısın. Sadece verilen bağlam içindeki bilgileri kullanarak Türkçe cevap ver. Bağlamda olmayan bir şey sorulursa 'Bu bilgi belgede yok' de. Kısa ve öz cevap ver."},
                        {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {soru}"}
                    ],
                    frequency_penalty=1.5,
                    presence_penalty=1.0,
                    max_tokens=500
                    )
                cevap = response.choices[0].message.content
            st.write(cevap)
            with st.expander("📎 Kaynaklar"):
                for i, src in enumerate(sources):
                    st.caption(f"Parça {i+1}: {src[:200]}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": cevap,
            "sources": sources
        })