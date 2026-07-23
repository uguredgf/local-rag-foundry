import sys
import subprocess
import re
import chromadb
from pypdf import PdfReader
from openai import OpenAI
from foundry_local_sdk import Configuration, FoundryLocalManager

sys.path.append("src")
from chunker import sentence_chunker, deduplicate_chunks

# SDK ile servisi başlat ve port al
config = Configuration(app_name="local_rag")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

# Port bul
result = subprocess.run(["foundry", "service", "status"], capture_output=True, text=True)
match = re.search(r'http://127\.0\.0\.1:(\d+)', result.stdout)
port = match.group(1) if match else "51250"
print(f"Foundry port: {port}")

# OpenAI compat. API ile bağlan
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="foundry")

# ChromaDB
chroma = chromadb.Client()
try:
    chroma.delete_collection("sdk_collection")
except:
    pass
collection = chroma.create_collection("sdk_collection")

# PDF yükle
reader = PdfReader("data/makale.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() or ""

chunks = sentence_chunker(full_text)
chunks = deduplicate_chunks(chunks)
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"{len(chunks)} parça yüklendi.")

# Soru-cevap döngüsü
print("\nSorunuzu yazın (çıkmak için 'exit'):\n")
while True:
    soru = input("Soru: ")
    if soru.lower() == "exit":
        break

    results = collection.query(query_texts=[soru], n_results=3)
    context = "\n".join(results['documents'][0])

    response = client.chat.completions.create(
        model="Phi-4-mini-instruct-cuda-gpu:5",
        messages=[
            {"role": "system", "content": "Sen bir akademik makale asistanısın. Sadece verilen bağlamdaki bilgileri kullanarak Türkçe cevap ver. Bağlamda yoksa 'Bu bilgi belgede yok' de. Kısa ve öz cevap ver."},
            {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {soru}"}
        ],
        max_tokens=250
    )
    print(f"\nCevap: {response.choices[0].message.content}\n")