from rank_bm25 import BM25Okapi
import re

class HybridSearch:
    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def index(self, chunks):
        """Chunk'ları indexle"""
        self.chunks = chunks
        tokenized = [self._tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text):
        """Metni kelimelere böl"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def keyword_search(self, query, n=5):
        """BM25 keyword arama"""
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.chunks[i], scores[i], i) for i in top_indices]

    def hybrid_search(self, query, vector_results, n=3):
        """
        Vektör sonuçları + BM25 sonuçlarını birleştir
        vector_results: ChromaDB'den gelen liste
        """
        # BM25 sonuçları
        keyword_results = self.keyword_search(query, n=10)
        keyword_chunks = [r[0] for r in keyword_results]
        keyword_scores = {r[0]: r[1] for r in keyword_results}

        # Vektör sonuçları normalize et
        max_kw_score = max([r[1] for r in keyword_results]) if keyword_results else 1

        # Her chunk için skor hesapla
        combined = {}

        # Vektör sonuçlarına skor ver (sırasına göre)
        for i, chunk in enumerate(vector_results):
            vector_score = 1.0 / (i + 1)  # 1, 0.5, 0.33...
            combined[chunk] = combined.get(chunk, 0) + vector_score * 0.6

        # Keyword sonuçlarına skor ver
        for chunk, score, _ in keyword_results:
            norm_score = score / max_kw_score if max_kw_score > 0 else 0
            combined[chunk] = combined.get(chunk, 0) + norm_score * 0.4

        # Sırala
        sorted_chunks = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in sorted_chunks[:n]]