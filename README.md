# 🤖 Local RAG with Foundry Local

Tamamen lokalde çalışan RAG (Retrieval-Augmented Generation) uygulaması.
İnternet bağlantısı gerekmez, API ücreti yoktur, verileriniz cihazınızdan çıkmaz.

## 🎯 Proje Hakkında

Bu proje, Microsoft Foundry Local kullanarak PDF dokümanlarını sorgulayan
bir yapay zeka uygulaması geliştirmeyi amaçlamaktadır. Kullanıcılar PDF
yükleyip, belgeden kaynaklanan sorular sorabilir.

## 🏗️ Mimari

PDF → Chunking → ChromaDB → Retrieval → Phi-4-mini → Cevap
(Vektör DB)              (Foundry Local)

## 🛠️ Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| LLM | Microsoft Phi-4-mini (Foundry Local) |
| Vektör Veritabanı | ChromaDB |
| Embedding | all-MiniLM-L6-v2 |
| Arayüz | Streamlit |
| Değerlendirme | Özel metrikler (Faithfulness, Relevancy, Precision) |

## 🚀 Kurulum

### 1. Foundry Local Kur
```powershell
winget install Microsoft.FoundryLocal
foundry model load Phi-4-mini-instruct-cuda-gpu:5
```

### 2. Repoyu Klonla
```powershell
git clone https://github.com/KULLANICI_ADIN/local-rag-foundry.git
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
│   ├── app.py          # Streamlit arayüzü
│   ├── chunker.py      # Chunking stratejileri
│   ├── rag.py          # Terminal tabanlı RAG
│   └── ingest.py       # PDF işleme
├── data/
│   └── sample_docs/    # Örnek dokümanlar
├── eval/
│   ├── evaluate.py     # Değerlendirme scripti
│   └── ragas_results.json  # Sonuçlar
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
- Fixed-size chunking cümle ortasında keserek LLM'in halüsinasyon üretmesine yol açar
- Çoklu PDF desteği ile farklı belgelerden bilgi birleştirilebilir
- Port otomatik algılama sayesinde her oturumda manuel ayar gerekmez

## 🎥 Demo Video

[Yakında eklenecek]