from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        print("Re-ranker yükleniyor...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Re-ranker hazır!")

    def rerank(self, query, chunks, top_n=3):
        """
        Verilen chunk'ları soruyla karşılaştırıp yeniden sırala
        """
        if not chunks:
            return chunks

        # Her chunk için (soru, chunk) çifti oluştur
        pairs = [(query, chunk) for chunk in chunks]

        # Puanları hesapla
        scores = self.model.predict(pairs)

        # Puana göre sırala
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # En iyi top_n tanesini döndür
        return [chunk for chunk, score in ranked[:top_n]]