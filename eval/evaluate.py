import sys
import subprocess
import re
import chromadb
import json
from pypdf import PdfReader
from openai import OpenAI

sys.path.append("src")
from chunker import sentence_chunker

def get_foundry_port():
    try:
        result = subprocess.run(["foundry", "service", "status"], capture_output=True, text=True)
        match = re.search(r'http://127\.0\.0\.1:(\d+)', result.stdout)
        if match:
            return match.group(1)
    except:
        pass
    return "51250"

port = get_foundry_port()
print(f"Foundry port: {port}")
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="foundry")

# PDF yükle
reader = PdfReader("data/makale.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() or ""

chunks = sentence_chunker(full_text)
chroma = chromadb.Client()
try:
    chroma.delete_collection("eval_collection")
except:
    pass
collection = chroma.create_collection("eval_collection")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"{len(chunks)} parça yüklendi.\n")

# --- Metrikler ---

def faithfulness_score(answer, contexts):
    """Cevaptaki kelimelerin kaçı context'ten geliyor?"""
    answer_words = set(answer.lower().split())
    context_words = set(" ".join(contexts).lower().split())
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return round(len(overlap) / len(answer_words), 3)

def relevancy_score(question, answer):
    """Soru kelimeleri cevap içinde var mı?"""
    q_words = set(question.lower().split()) - {"what","is","are","the","how","does","a","an","of"}
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.0
    overlap = q_words & a_words
    return round(len(overlap) / len(q_words), 3)

def context_precision_score(question, contexts):
    """Context parçaları soruyla ne kadar alakalı?"""
    q_words = set(question.lower().split())
    scores = []
    for ctx in contexts:
        ctx_words = set(ctx.lower().split())
        if not ctx_words:
            continue
        overlap = q_words & ctx_words
        scores.append(len(overlap) / len(q_words))
    return round(sum(scores) / len(scores), 3) if scores else 0.0

# --- Test soruları ---
test_questions = [
    "What is artificial intelligence?",
    "What are the main applications of AI?",
    "What is machine learning?",
    "How does deep learning work?",
    "What are the challenges of AI?",
]

results = []
print("Sorular işleniyor...\n")

for q in test_questions:
    retrieved = collection.query(query_texts=[q], n_results=3)
    context_list = retrieved['documents'][0]
    context = "\n".join(context_list)

    response = client.chat.completions.create(
        model="Phi-4-mini-instruct-cuda-gpu:5",
        messages=[
            {"role": "system", "content": "Answer based only on the given context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
        ],
        frequency_penalty=1.5,
        presence_penalty=1.0,
        max_tokens=300
    )
    answer = response.choices[0].message.content

    faith = faithfulness_score(answer, context_list)
    relev = relevancy_score(q, answer)
    prec  = context_precision_score(q, context_list)

    results.append({
        "question": q,
        "answer": answer[:200],
        "faithfulness": faith,
        "answer_relevancy": relev,
        "context_precision": prec,
    })

    print(f"Soru: {q}")
    print(f"  Faithfulness:      {faith}")
    print(f"  Answer Relevancy:  {relev}")
    print(f"  Context Precision: {prec}\n")

# Ortalama
avg_faith = round(sum(r["faithfulness"] for r in results) / len(results), 3)
avg_relev = round(sum(r["answer_relevancy"] for r in results) / len(results), 3)
avg_prec  = round(sum(r["context_precision"] for r in results) / len(results), 3)

print("=== ORTALAMA SONUÇLAR ===")
print(f"Faithfulness:      {avg_faith}")
print(f"Answer Relevancy:  {avg_relev}")
print(f"Context Precision: {avg_prec}")

output = {
    "results": results,
    "averages": {
        "faithfulness": avg_faith,
        "answer_relevancy": avg_relev,
        "context_precision": avg_prec,
    }
}

with open("eval/ragas_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\nSonuçlar eval/ragas_results.json dosyasına kaydedildi.")