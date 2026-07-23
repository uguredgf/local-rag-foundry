# 🤖 Local RAG with Foundry Local

Tamamen lokalde çalışan gelişmiş RAG (Retrieval-Augmented Generation) uygulaması.
İnternet bağlantısı gerekmez, API ücreti yoktur, verileriniz cihazınızdan çıkmaz.

## 🎯 Proje Hakkında

Bu proje, Microsoft Foundry Local kullanarak PDF dokümanlarını sorgulayan
bir yapay zeka uygulaması geliştirmeyi amaçlamaktadır. Kullanıcılar PDF
yükleyip belgeden kaynaklanan sorular sorabilir. Sistem tamamen lokalde
çalıştığından kurumsal ve akademik kullanım için idealdir.

## 🏗️ Mimari
PDF → Chunking → ChromaDB → Query Expansion → Hybrid Search → Re-ranker → Phi-4-mini → Cevap

## 🔄 Pipeline Detayı

1. **Chunking** — PDF sentence-based veya fixed-size parçalara bölünür
2. **Embedding** — ChromaDB all-MiniLM-L6-v2 ile vektöre çevrilir
3. **Query Expansion** — Soru LLM tarafından 3 farklı şekilde genişletilir
4. **Hybrid Search** — Vektör araması + BM25 keyword araması birleştirilir
5. **Re-ranking** — CrossEncoder ile parçalar yeniden sıralanır
6. **Generation** — Phi-4-mini bağlam kullanarak Türkçe cevap üretir

## 🛠️ Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| LLM | Microsoft Phi-4-mini (Foundry Local) |
| Vektör Veritabanı | ChromaDB |
| Embedding | all-MiniLM-L6-v2 |
| Keyword Arama | BM25 (rank-bm25) |
| Re-ranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Arayüz | Streamlit |
| SDK | Foundry Local SDK |

## 🚀 Kurulum

### 1. Foundry Local Kur
```powershell
winget install Microsoft.FoundryLocal
foundry model load Phi-4-mini-instruct-cuda-gpu:5
```

### 2. Repoyu Klonla
```powershell
git clone https://github.com/uguredgf/local-rag-foundry.git
cd local-rag-foundry
```

### 3. Bağımlılıkları Kur
```powershell
pip install -r requirements.txt
```

### 4. Uygulamayı Başlat
```powershell
foundry service start
streamlit run src/app.py
```

## 📁 Proje Yapısı
/
├── src/
│ ├── app.py # Streamlit arayüzü
│ ├── chunker.py # Chunking stratejileri
│ ├── hybrid_search.py # Hybrid Search (vektör + BM25)
│ ├── reranker.py # CrossEncoder re-ranking
│ ├── query_expander.py # Query expansion
│ ├── rag.py # Terminal tabanlı RAG
│ └── ingest.py # PDF işleme
├── eval/
│ ├── evaluate.py # Değerlendirme scripti
│ └── ragas_results.json # Sonuçlar
├── data/
│ └── sample_docs/ # Örnek dokümanlar
├── requirements.txt
└── README.md

## 📊 Değerlendirme Sonuçları

| Metrik | Sonuç |
|---|---|
| Faithfulness | 0.044 |
| Answer Relevancy | 0.467 |
| Context Precision | 0.277 |

> **Not:** Faithfulness değeri düşük görünmektedir çünkü context İngilizce,
> cevaplar Türkçe üretilmiştir. Kelime örtüşme bazlı metrik bu durumda
> dil farkından dolayı düşük sonuç vermektedir.

## 🔍 Temel Bulgular

- **Sentence-based chunking**, fixed-size'a göre daha kaliteli cevaplar üretir
- Fixed-size chunking cümle ortasında keserek LLM halüsinasyon üretmesine yol açar
- **Hybrid Search**, sadece vektör aramasına göre spesifik teknik terimleri daha iyi bulur
- **Re-ranking** ile retrieval kalitesi artmaktadır
- **Query Expansion** ile tek sorgu yerine çoklu sorgu daha kapsamlı sonuç verir
- Türkçe sorular otomatik İngilizce'ye çevrilerek İngilizce belgelerde arama yapılabilir

## 🎥 Demo Video

[Yakında eklenecek]