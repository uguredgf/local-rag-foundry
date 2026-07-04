from pypdf import PdfReader
import chromadb
from openai import OpenAI
import subprocess
import re

# Foundry Local'e bağlan
def get_foundry_port():
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            capture_output=True, text=True
        )
        match = re.search(r'http://127\.0\.0\.1:(\d+)', result.stdout)
        if match:
            return match.group(1)
    except:
        pass
    return "51250"

port = get_foundry_port()
client = OpenAI(
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="foundry"
)

# PDF'i oku ve ChromaDB'ye kaydet
print("PDF yükleniyor...")
reader = PdfReader("data/makale.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(full_text)

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("makale")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"{len(chunks)} parça yüklendi.\n")

# Soru-cevap döngüsü
print("Sorunuzu yazın (çıkmak için 'exit'):\n")
while True:
    soru = input("Soru: ")
    if soru.lower() == "exit":
        break

    # İlgili parçaları bul
    results = collection.query(query_texts=[soru], n_results=3)
    context = "\n".join(results['documents'][0])

    # LLM'e gönder
    response = client.chat.completions.create(
        model="Phi-4-mini-instruct-cuda-gpu:5",
        messages=[
            {"role": "system", "content": "Sen bir akademik makale asistanısın. Sadece verilen bağlam içindeki bilgileri kullanarak Türkçe cevap ver."},
            {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {soru}"}
        ]
    )
    print(f"\nCevap: {response.choices[0].message.content}\n")