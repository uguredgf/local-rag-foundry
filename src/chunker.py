import re

def fixed_chunker(text, chunk_size=500, overlap=50):
    """Basit karakter bazlı bölme"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def sentence_chunker(text, sentences_per_chunk=5, overlap=1):
    """Cümle bazlı bölme — cümle ortasında kesmez"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def deduplicate_chunks(chunks, threshold=0.9):
    """Çok benzer parçaları filtrele"""
    unique = []
    for chunk in chunks:
        is_duplicate = False
        for u in unique:
            # Basit benzerlik kontrolü
            words_chunk = set(chunk.lower().split())
            words_u = set(u.lower().split())
            if len(words_chunk) == 0:
                continue
            similarity = len(words_chunk & words_u) / len(words_chunk | words_u)
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(chunk)
    return unique