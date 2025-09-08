# test_vector_search.py
import asyncio
from main import save_embedding, find_similar_queries

async def test_vector_search():
    # Insert sample queries
    sample_queries = [
        "I have a headache and fever",
        "Chest pain and shortness of breath",
        "Sore throat and cough"
    ]
    for query in sample_queries:
        save_embedding("test_session", "test_user", "query", query)
    
    # Test vector search
    results = await find_similar_queries("test_session", "fever and headache")
    print("Similar queries:", results)

if __name__ == "__main__":
    asyncio.run(test_vector_search())