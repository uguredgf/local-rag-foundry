from pypdf import PdfReader
import chromadb

print("PDF okunuyor...")
reader = PdfReader("data/makale.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print(f"Toplam karakter: {len(full_text)}")

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(full_text)
print(f"Toplam parça sayısı: {len(chunks)}")

client = chromadb.Client()
collection = client.create_collection("makale")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )

print("Tüm parçalar ChromaDB'ye kaydedildi!")

results = collection.query(
    query_texts=["What is artificial intelligence?"],
    n_results=2
)

print("\n--- TEST SORGUSU SONUCU ---")
for doc in results['documents'][0]:
    print(doc[:200])
    print("---")