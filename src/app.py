import streamlit as st
import chromadb
from pypdf import PdfReader
from openai import OpenAI
import tempfile
import os
import subprocess
import re
from chunker import fixed_chunker, sentence_chunker, deduplicate_chunks
from hybrid_search import HybridSearch
from reranker import Reranker

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
if "hybrid_searcher" not in st.session_state:
    st.session_state.hybrid_searcher = HybridSearch()
if "reranker" not in st.session_state:
    st.session_state.reranker = Reranker()

def split_text(text, strategy, chunk_size=500, overlap=50):
    if strategy == "Sentence-based (Önerilen)":
        return sentence_chunker(text)
    return fixed_chunker(text, chunk_size, overlap)

with st.sidebar:
    st.header("📄 Doküman Yükle")
    
    st.subheader("⚙️ Ayarlar")
    chunk_strategy = st.selectbox(
        "Chunking Stratejisi",
        ["Sentence-based (Önerilen)", "Fixed-size"],
    )
    
    # Konuşma geçmişi uzunluğu
    history_length = st.slider(
        "Konuşma geçmişi (kaç önceki mesaj)",
        min_value=0, max_value=10, value=3,
        help="0 = geçmiş yok, 10 = son 10 mesajı hatırlar"
    )

    uploaded_files = st.file_uploader(
        "PDF seç (birden fazla olabilir)",
        type="pdf",
        accept_multiple_files=True
    )

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

                chunks = split_text(full_text, chunk_strategy)
                all_chunks.extend(chunks)
                loaded_names.append(uploaded_file.name)

        if all_chunks:
            all_chunks = deduplicate_chunks(all_chunks)
            collection_name = "rag_collection"
            try:
                chroma.delete_collection(collection_name)
            except:
                pass
            collection = chroma.create_collection(collection_name)
            collection.add(
                documents=all_chunks,
                ids=[f"chunk_{i}" for i in range(len(all_chunks))]
            )
            st.session_state.collection = collection
            st.session_state.pdf_names = loaded_names
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.hybrid_searcher = HybridSearch()
            st.session_state.hybrid_searcher.index(all_chunks)

            # Otomatik özet
            with st.spinner("Özet çıkarılıyor..."):
                ilk_chunks = " ".join(all_chunks[:5])
                ozet_response = client.chat.completions.create(
                    model="Phi-4-mini-instruct-cuda-gpu:5",
                    messages=[
                        {"role": "system", "content": "Verilen metni 3 cümleyle Türkçe özetle."},
                        {"role": "user", "content": ilk_chunks}
                    ],
                    max_tokens=200
                )
                st.session_state.ozet = ozet_response.choices[0].message.content

            st.success(f"✅ {len(all_chunks)} parça yüklendi!")

    if st.session_state.pdf_names:
        if st.session_state.ozet:
            st.divider()
            st.markdown("**📝 Belge Özeti:**")
            st.caption(st.session_state.ozet)
        st.divider()
        st.markdown("**Yüklü Dokümanlar:**")
        for name in st.session_state.pdf_names:
            st.caption(f"📖 {name}")

    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        if "ozet" not in st.session_state:
            st.session_state.ozet = None
        st.rerun()

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
                # Son soruyu da bağlam olarak ekle
                son_soru = ""
                if len(st.session_state.chat_history) >= 2:
                    son_soru = f"Previous question: {st.session_state.chat_history[-2]['content']}\n"

                translation = client.chat.completions.create(
                    model="Phi-4-mini-instruct-cuda-gpu:5",
                    messages=[
                        {"role": "system", "content": "Translate the following question to English. If the question refers to something from the previous question, make the reference explicit. Return only the translation, nothing else."},
                        {"role": "user", "content": f"{son_soru}Current question: {soru}"}
                    ],
                    max_tokens=100
                )
                soru_en = translation.choices[0].message.content.strip()

                results = st.session_state.collection.query(
                    query_texts=[soru_en], n_results=5
                )
                vector_results = results['documents'][0]

                # Hybrid search
                hybrid_results = st.session_state.hybrid_searcher.hybrid_search(
                    query=soru_en,
                    vector_results=vector_results,
                    n=5
                )

                # Re-ranking
                context_list = st.session_state.reranker.rerank(
                    query=soru_en,
                    chunks=hybrid_results,
                    top_n=3
                )
                context = "\n".join(context_list)

                # Konuşma geçmişini hazırla
                system_msg = {
                    "role": "system",
                    "content": "Sen bir akademik makale asistanısın. SADECE verilen bağlamdaki bilgileri kullan. Bağlamda yoksa sadece 'Bu bilgi belgede yok' de, başka hiçbir şey ekleme. Maksimum 3 cümle ile cevap ver. Asla ek yorum, link veya öneri ekleme."
                }

                # Son N mesajı al
                recent_history = st.session_state.chat_history[-history_length*2:] if history_length > 0 else []

                # Şu anki soruyu bağlamla birlikte ekle
                current_msg = {
                    "role": "user",
                    "content": f"Bağlam:\n{context}\n\nSoru: {soru}"
                }

                messages_to_send = [system_msg] + recent_history + [current_msg]

                response = client.chat.completions.create(
                    model="Phi-4-mini-instruct-cuda-gpu:5",
                    messages=messages_to_send,
                    max_tokens=250
                )
                cevap = response.choices[0].message.content

            st.write(cevap)
            with st.expander("📎 Kaynaklar"):
                for i, src in enumerate(context_list):
                    st.caption(f"Parça {i+1}: {src[:200]}...")

        # Geçmişe ekle
        st.session_state.chat_history.append({"role": "user", "content": soru})
        st.session_state.chat_history.append({"role": "assistant", "content": cevap})

        st.session_state.messages.append({
            "role": "assistant",
            "content": cevap,
            "sources": context_list
        })