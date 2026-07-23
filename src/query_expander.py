def expand_query(client, model, query):
    """
    Soruyu genişlet — daha iyi retrieval için
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a search query expander. 
Given a question, generate 3 alternative search queries that would help find relevant information.
Return ONLY the queries, one per line, nothing else."""},
            {"role": "user", "content": f"Original query: {query}"}
        ],
        max_tokens=100
    )
    
    expanded = response.choices[0].message.content.strip()
    queries = [query] + [q.strip() for q in expanded.split("\n") if q.strip()]
    return queries[:4]  # Orijinal + max 3 genişletilmiş