# # # # backend/utils.py
# # # import json
# # # import logging
# # # import pymysql
# # # from typing import List, Dict, Any

# # # logger = logging.getLogger(__name__)

# # # # Database configuration (main.py se copy karo)
# # # DB_CONFIG = {
# # #     "host": "gateway01.us-west-2.prod.aws.tidbcloud.com",
# # #     "port": 4000,
# # #     "user": "34oY1b3G6arXWAM.root",
# # #     "password": "M9iWYjgizxiiT1qh",
# # #     "database": "test",
# # #     "charset": "utf8mb4",
# # #     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}
# # # }

# # # def get_db():
# # #     """Establish a connection to TiDB."""
# # #     try:
# # #         connection = pymysql.connect(**DB_CONFIG)
# # #         return connection
# # #     except pymysql.err.OperationalError as e:
# # #         logger.error(f"Failed to connect to TiDB: {str(e)}")
# # #         raise Exception(f"Database connection failed: {str(e)}")

# # # def fallback_text_search(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# # #     """Fallback text search if vector search fails."""
# # #     conn = get_db()
# # #     try:
# # #         with conn.cursor() as cur:
# # #             cur.execute("""
# # #                 SELECT id, content, metadata
# # #                 FROM specialist_vectors 
# # #                 WHERE specialty = %s 
# # #                 AND content LIKE %s
# # #                 LIMIT %s
# # #             """, (specialty, f"%{query}%", top_k))
            
# # #             results = cur.fetchall()
# # #             return [{"id": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]
# # #     except Exception as e:
# # #         logger.error(f"Fallback search error: {str(e)}")
# # #         return []
# # #     finally:
# # #         conn.close()

# # # def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# # #     """Search similar cases using vector similarity search with TiDB's <-> operator."""
# # #     # Import generate_embedding yahan par karo taake circular import na ho
# # #     from main import generate_embedding
    
# # #     conn = get_db()
# # #     try:
# # #         # Generate embedding for the query
# # #         query_embedding = generate_embedding(query)
        
# # #         with conn.cursor() as cur:
# # #             # Use TiDB's vector similarity search with <-> operator
# # #             cur.execute("""
# # #                 SELECT 
# # #                     id, 
# # #                     content, 
# # #                     metadata,
# # #                     embedding <-> %s as distance
# # #                 FROM specialist_vectors 
# # #                 WHERE specialty = %s 
# # #                 ORDER BY distance ASC
# # #                 LIMIT %s
# # #             """, (json.dumps(query_embedding), specialty, top_k))
            
# # #             results = cur.fetchall()
# # #             return [{
# # #                 "id": r[0], 
# # #                 "content": r[1], 
# # #                 "metadata": json.loads(r[2]) if r[2] else {},
# # #                 "similarity_score": float(1 - r[3])  # Convert distance to similarity
# # #             } for r in results]
# # #     except Exception as e:
# # #         logger.error(f"Vector search error: {str(e)}")
# # #         # Fallback to text search
# # #         return fallback_text_search(query, specialty, top_k)
# # #     finally:
# # #         conn.close()

# # # backend/utils.py
# # import json
# # import logging
# # import pymysql
# # from typing import List, Dict, Any
# # import google.generativeai as genai
# # import os

# # logger = logging.getLogger(__name__)

# # # Database configuration
# # DB_CONFIG = {
# #     "host": "gateway01.us-west-2.prod.aws.tidbcloud.com",
# #     "port": 4000,
# #     "user": "34oY1b3G6arXWAM.root",
# #     "password": "M9iWYjgizxiiT1qh",
# #     "database": "test",
# #     "charset": "utf8mb4",
# #     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}
# # }

# # def get_db():
# #     """Establish a connection to TiDB."""
# #     try:
# #         connection = pymysql.connect(**DB_CONFIG)
# #         return connection
# #     except pymysql.err.OperationalError as e:
# #         logger.error(f"Failed to connect to TiDB: {str(e)}")
# #         raise Exception(f"Database connection failed: {str(e)}")

# # def generate_embedding(text: str) -> List[float]:
# #     """Generate a real embedding vector using the Gemini embedding model."""
# #     try:
# #         # Configure Gemini
# #         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
# #         # Call the Gemini Embedding API
# #         result = genai.embed_content(
# #             model="models/embedding-001",
# #             content=text,
# #             task_type="retrieval_document"
# #         )
# #         return result['embedding']
# #     except Exception as e:
# #         logger.error(f"Embedding generation failed: {str(e)}")
# #         # Fallback to avoid breaking the application
# #         return [0.0] * 768

# # def fallback_text_search(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# #     """Fallback text search if vector search fails."""
# #     conn = get_db()
# #     try:
# #         with conn.cursor() as cur:
# #             cur.execute("""
# #                 SELECT id, content, metadata
# #                 FROM specialist_vectors 
# #                 WHERE specialty = %s 
# #                 AND content LIKE %s
# #                 LIMIT %s
# #             """, (specialty, f"%{query}%", top_k))
            
# #             results = cur.fetchall()
# #             return [{"id": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]
# #     except Exception as e:
# #         logger.error(f"Fallback search error: {str(e)}")
# #         return []
# #     finally:
# #         conn.close()

# # def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# #     """Search similar cases using vector similarity search with TiDB's <-> operator."""
# #     conn = get_db()
# #     try:
# #         # Generate embedding for the query
# #         query_embedding = generate_embedding(query)
        
# #         with conn.cursor() as cur:
# #             # Use TiDB's vector similarity search with <-> operator
# #             cur.execute("""
# #                 SELECT 
# #                     id, 
# #                     content, 
# #                     metadata,
# #                     embedding <-> %s as distance
# #                 FROM specialist_vectors 
# #                 WHERE specialty = %s 
# #                 ORDER BY distance ASC
# #                 LIMIT %s
# #             """, (json.dumps(query_embedding), specialty, top_k))
            
# #             results = cur.fetchall()
# #             return [{
# #                 "id": r[0], 
# #                 "content": r[1], 
# #                 "metadata": json.loads(r[2]) if r[2] else {},
# #                 "similarity_score": float(1 - r[3])  # Convert distance to similarity
# #             } for r in results]
# #     except Exception as e:
# #         logger.error(f"Vector search error: {str(e)}")
# #         # Fallback to text search
# #         return fallback_text_search(query, specialty, top_k)
# #     finally:
# #         conn.close()


# # backend/utils.py
# import json
# import logging
# import pymysql
# from typing import List, Dict, Any
# import google.generativeai as genai
# import os

# logger = logging.getLogger(__name__)

# # Database configuration
# DB_CONFIG = {
#     "host": "gateway01.us-west-2.prod.aws.tidbcloud.com",
#     "port": 4000,
#     "user": "34oY1b3G6arXWAM.root",
#     "password": "M9iWYjgizxiiT1qh",
#     "database": "test",
#     "charset": "utf8mb4",
#     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}
# }

# def get_db():
#     """Establish a connection to TiDB."""
#     try:
#         connection = pymysql.connect(**DB_CONFIG)
#         return connection
#     except pymysql.err.OperationalError as e:
#         logger.error(f"Failed to connect to TiDB: {str(e)}")
#         raise Exception(f"Database connection failed: {str(e)}")

# def generate_embedding(text: str) -> List[float]:
#     """Generate a real embedding vector using the Gemini embedding model."""
#     try:
#         # Configure Gemini
#         genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
#         # Call the Gemini Embedding API
#         result = genai.embed_content(
#             model="models/embedding-001",
#             content=text,
#             task_type="retrieval_document"
#         )
#         return result['embedding']
#     except Exception as e:
#         logger.error(f"Embedding generation failed: {str(e)}")
#         # Fallback to avoid breaking the application
#         return [0.0] * 768

# def fallback_text_search(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
#     """Fallback text search if vector search fails."""
#     conn = get_db()
#     try:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 SELECT id, content, metadata
#                 FROM specialist_vectors 
#                 WHERE specialty = %s 
#                 AND content LIKE %s
#                 LIMIT %s
#             """, (specialty, f"%{query}%", top_k))
            
#             results = cur.fetchall()
#             return [{"id": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]
#     except Exception as e:
#         logger.error(f"Fallback search error: {str(e)}")
#         return []
#     finally:
#         conn.close()

# def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
#     """Search similar cases using vector similarity search with TiDB's <-> operator."""
#     conn = get_db()
#     try:
#         # Generate embedding for the query
#         query_embedding = generate_embedding(query)
        
#         # Convert the embedding to a string representation that TiDB expects
#         embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
#         with conn.cursor() as cur:
#             # Use TiDB's vector similarity search with <-> operator
#             cur.execute("""
#                 SELECT 
#                     id, 
#                     content, 
#                     metadata,
#                     embedding <-> %s as distance
#                 FROM specialist_vectors 
#                 WHERE specialty = %s 
#                 ORDER BY distance ASC
#                 LIMIT %s
#             """, (embedding_str, specialty, top_k))
            
#             results = cur.fetchall()
#             return [{
#                 "id": r[0], 
#                 "content": r[1], 
#                 "metadata": json.loads(r[2]) if r[2] else {},
#                 "similarity_score": float(1 - r[3])  # Convert distance to similarity
#             } for r in results]
#     except Exception as e:
#         logger.error(f"Vector search error: {str(e)}")
#         # Fallback to text search
#         return fallback_text_search(query, specialty, top_k)
#     finally:
#         conn.close()


# backend/utils.py
import json
import logging
import pymysql
from typing import List, Dict, Any
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "gateway01.us-west-2.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "34oY1b3G6arXWAM.root",
    "password": "M9iWYjgizxiiT1qh",
    "database": "test",
    "charset": "utf8mb4",
    "ssl": {"ssl_mode": "VERIFY_IDENTITY"}
}

def get_db():
    """Establish a connection to TiDB."""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except pymysql.err.OperationalError as e:
        logger.error(f"Failed to connect to TiDB: {str(e)}")
        raise Exception(f"Database connection failed: {str(e)}")

def generate_embedding(text: str) -> List[float]:
    """Generate a real embedding vector using the Gemini embedding model."""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        # Fallback to avoid breaking the application
        return [0.0] * 768

def fallback_text_search(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
    """Fallback text search if vector search fails."""
    conn = get_db()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id, content, metadata
                FROM specialist_vectors 
                WHERE specialty = %s 
                AND content LIKE %s
                LIMIT %s
            """, (specialty, f"%{query}%", top_k))
            
            results = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "similarity_score": 0.0  # Fallback has no real similarity
                }
                for row in results
            ]
    except Exception as e:
        logger.error(f"Fallback search error: {str(e)}")
        return []
    finally:
        conn.close()

def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
    """
    Search similar cases using vector similarity search with TiDB's <-> operator.
    """
    conn = get_db()
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)

        # Convert the embedding to a string representation that TiDB expects
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            sql_query = """
                SELECT 
                    id, 
                    content, 
                    metadata,
                    embedding <-> %s AS distance
                FROM specialist_vectors 
                WHERE specialty = %s 
                ORDER BY distance ASC
                LIMIT %s
            """
            logger.debug(f"Executing vector search SQL: {sql_query}")
            cur.execute(sql_query, (embedding_str, specialty, top_k))

            results = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    # Normalize distance: closer = higher similarity
                    "similarity_score": float(1 / (1 + row["distance"])) if row["distance"] is not None else 0.0
                }
                for row in results
            ]
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        # Fallback to text search if vector search fails
        return fallback_text_search(query, specialty, top_k)
    finally:
        conn.close()
