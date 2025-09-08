# # # # # correct code 
# # # # import os
# # # # import json
# # # # import asyncio
# # # # from datetime import datetime
# # # # from fastapi import FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from fastapi.responses import JSONResponse
# # # # from pydantic import BaseModel, Field
# # # # from dotenv import load_dotenv
# # # # from typing import Optional, Dict, List, Any
# # # # import logging
# # # # from contextlib import asynccontextmanager
# # # # import uuid
# # # # import httpx
# # # # import re
# # # # import pymysql

# # # # # Import OpenAI Agents SDK components (assumed to be available)
# # # # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # # # # Import agent creation functions
# # # # from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
# # # # from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
# # # # from medicura_agents.general_health_agent import create_general_health_agent
# # # # from medicura_agents.medical_term_agent import create_medical_term_agent
# # # # from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
# # # # from medicura_agents.about_agent import create_about_agent


# # # # # Import agent creation functions from specialist_agents
# # # # from specialist_agents.cardiology_ai import create_cardiology_agent
# # # # from specialist_agents.dermatology_ai import create_dermatology_agent
# # # # from specialist_agents.neurology_ai import create_neurology_agent
# # # # from specialist_agents.pulmonology_ai import create_pulmonology_agent
# # # # from specialist_agents.ophthalmology_ai import create_ophthalmology_agent
# # # # from specialist_agents.dental_ai import create_dental_agent
# # # # from specialist_agents.allergy_immunology_ai import create_allergy_immunology_agent
# # # # from specialist_agents.pediatrics_ai import create_pediatrics_agent
# # # # from specialist_agents.orthopedics_ai import create_orthopedics_agent
# # # # from specialist_agents.mental_health_ai import create_mental_health_agent
# # # # from specialist_agents.endocrinology_ai import create_endocrinology_agent
# # # # from specialist_agents.gastroenterology_ai import create_gastroenterology_agent
# # # # from specialist_agents.radiology_ai import create_radiology_agent
# # # # from specialist_agents.infectious_disease_ai import create_infectious_disease_agent
# # # # from specialist_agents.vaccination_advisor_ai import create_vaccination_advisor_agent


# # # # # Configure logging
# # # # logging.basicConfig(level=logging.INFO)
# # # # logger = logging.getLogger(__name__)

# # # # # Load environment variables from .env file
# # # # load_dotenv()

# # # # # TiDB Configuration with minimal SSL
# # # # DB_CONFIG = {
# # # #     "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
# # # #     "port": 4000,
# # # #     "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
# # # #     "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
# # # #     "database": os.getenv("TIDB_DATABASE", "test"),
# # # #     "charset": "utf8mb4",
# # # #     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
# # # # }

# # # # def get_db():
# # # #     """Establish a connection to TiDB."""
# # # #     try:
# # # #         connection = pymysql.connect(**DB_CONFIG)
# # # #         return connection
# # # #     except pymysql.err.OperationalError as e:
# # # #         logger.error(f"Failed to connect to TiDB: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# # # # @asynccontextmanager
# # # # async def lifespan(app: FastAPI):
# # # #     """Manage application startup and shutdown."""
# # # #     logger.info("Starting Medicura-AI Health Assistant")
# # # #     logger.info("Connecting to TiDB...")
# # # #     try:
# # # #         conn = get_db()
# # # #         with conn.cursor() as cur:
# # # #             cur.execute("""
# # # #                 CREATE TABLE IF NOT EXISTS chat_sessions (
# # # #                     session_id VARCHAR(100) PRIMARY KEY,
# # # #                     history JSON NOT NULL,
# # # #                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# # # #                 )
# # # #             """)
# # # #             conn.commit()
# # # #         conn.close()
# # # #         logger.info("TiDB connected and table ready.")
# # # #         yield
# # # #     except Exception as e:
# # # #         logger.error(f"Lifespan error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Application startup failed")
# # # #     finally:
# # # #         logger.info("Shutting down Medicura-AI Health Assistant")

# # # # app = FastAPI(
# # # #     title="Medicura-AI Health Assistant",
# # # #     description="AI-powered health assistant for symptom analysis and medical queries",
# # # #     version="2.1.0",
# # # #     lifespan=lifespan,
# # # #     docs_url="/api/docs",
# # # #     redoc_url="/api/redoc"
# # # # )

# # # # # CORS Configuration
# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
# # # #     allow_credentials=True,
# # # #     allow_methods=["*"],
# # # #     allow_headers=["*"],
# # # # )

# # # # # Environment Variable Validation
# # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # if not GEMINI_API_KEY:
# # # #     logger.error("GEMINI_API_KEY not found in environment variables")
# # # #     raise ValueError("GEMINI_API_KEY environment variable is required")

# # # # TIDB_HOST = os.getenv("TIDB_HOST")
# # # # TIDB_USERNAME = os.getenv("TIDB_USERNAME")
# # # # TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
# # # # TIDB_DATABASE = os.getenv("TIDB_DATABASE")

# # # # # AI Agent Initialization
# # # # external_client = AsyncOpenAI(
# # # #     api_key=GEMINI_API_KEY,
# # # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # #     http_client=httpx.AsyncClient(timeout=60.0)
# # # # )

# # # # model = OpenAIChatCompletionsModel(
# # # #     model="gemini-2.0-flash",
# # # #     openai_client=external_client,
# # # # )

# # # # model_settings = ModelSettings(
# # # #     temperature=0.7,
# # # #     top_p=0.9,
# # # #     max_tokens=2048,
# # # # )

# # # # config = RunConfig(
# # # #     model=model,
# # # #     model_provider=external_client,
# # # #     model_settings=model_settings,
# # # #     tracing_disabled=True,
# # # # )

# # # # # Initialize Agents
# # # # symptom_analyzer_agent = create_symptom_analyzer_agent(model)
# # # # drug_interaction_agent = create_drug_interaction_agent(model)
# # # # general_health_agent = create_general_health_agent(model)
# # # # medical_term_agent = create_medical_term_agent(model)
# # # # report_analyzer_agent = create_report_analyzer_agent(model)
# # # # about_agent = create_about_agent(model)

# # # # def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
# # # #     """Extract JSON from agent response with multiple fallback methods."""
# # # #     try:
# # # #         # Method 1: Try direct JSON parsing
# # # #         try:
# # # #             return json.loads(response.strip())
# # # #         except json.JSONDecodeError:
# # # #             pass
        
# # # #         # Method 2: Extract from code blocks
# # # #         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
# # # #         if json_match:
# # # #             return json.loads(json_match.group(1))
        
# # # #         # Method 3: Find first JSON object
# # # #         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
# # # #         if brace_match:
# # # #             return json.loads(brace_match.group(0))
            
# # # #         # Method 4: Fallback structured response
# # # #         return {
# # # #             "summary": response,
# # # #             "detailed_analysis": "Detailed analysis based on your query",
# # # #             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
# # # #             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
# # # #             "type": "general"
# # # #         }
# # # #     except Exception as e:
# # # #         logger.warning(f"JSON extraction failed: {str(e)}")
# # # #         return None

# # # # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# # # #     """Run agent with enhanced thinking and robust error handling."""
# # # #     try:
# # # #         thinking_prompt = f"""
# # # #         USER QUERY: {prompt}
# # # #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
        
# # # #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# # # #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# # # #         """
        
# # # #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# # # #         parsed_response = extract_json_from_response(result.final_output)
        
# # # #         if parsed_response:
# # # #             parsed_response.update({
# # # #                 "timestamp": datetime.now().isoformat(),
# # # #                 "success": True,
# # # #                 "thinking_applied": True
# # # #             })
# # # #             return parsed_response
# # # #         else:
# # # #             return create_intelligent_response(result.final_output, prompt)
        
# # # #     except Exception as e:
# # # #         logger.error(f"Agent error: {str(e)}")
# # # #         return create_intelligent_response(f"Analysis of: {prompt}")

# # # # def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
# # # #     """Create a well-structured response from text."""
# # # #     return {
# # # #         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
# # # #         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
# # # #         "recommendations": [
# # # #             "Consult with a healthcare provider",
# # # #             "Provide complete medical history for assessment",
# # # #             "Follow evidence-based medical guidance"
# # # #         ],
# # # #         "when_to_seek_help": [
# # # #             "Immediately for severe or emergency symptoms",
# # # #             "Within 24-48 hours for persistent concerns",
# # # #             "Routinely for preventive care"
# # # #         ],
# # # #         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
# # # #         "type": "general",
# # # #         "timestamp": datetime.now().isoformat(),
# # # #         "success": True,
# # # #         "thinking_applied": True
# # # #     }

# # # # def load_history(session_id: str) -> List[dict]:
# # # #     """Load chat history from TiDB."""
# # # #     try:
# # # #         conn = get_db()
# # # #         with conn.cursor() as cur:
# # # #             cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
# # # #             result = cur.fetchone()
# # # #         conn.close()
# # # #         return json.loads(result[0]) if result else []
# # # #     except Exception as e:
# # # #         logger.error(f"Failed to load history: {str(e)}")
# # # #         return []

# # # # def save_history(session_id: str, history: List[dict]):
# # # #     """Save chat history to TiDB."""
# # # #     try:
# # # #         conn = get_db()
# # # #         with conn.cursor() as cur:
# # # #             cur.execute("""
# # # #                 INSERT INTO chat_sessions (session_id, history)
# # # #                 VALUES (%s, %s)
# # # #                 ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
# # # #             """, (session_id, json.dumps(history), json.dumps(history)))
# # # #             conn.commit()
# # # #         conn.close()
# # # #     except Exception as e:
# # # #         logger.error(f"Failed to save history: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Failed to save chat history")

# # # # # Pydantic Models
# # # # class ChatRequest(BaseModel):
# # # #     message: str = Field(..., min_length=1, max_length=1000)
# # # #     session_id: Optional[str] = Field(None, max_length=100)
# # # #     context: Optional[dict] = None

# # # # class DrugInteractionInput(BaseModel):
# # # #     medications: List[str] = Field(..., min_items=1, max_items=10)
# # # #     age: Optional[int] = Field(None, ge=0, le=120)
# # # #     gender: Optional[str] = Field(None, max_length=20)
# # # #     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
# # # #     other_medications: Optional[List[str]] = Field(None, max_items=20)

# # # # class MedicalTermInput(BaseModel):
# # # #     term: str = Field(..., min_length=1, max_length=100)
# # # #     language: Optional[str] = Field("en", max_length=10)

# # # # class ReportTextInput(BaseModel):
# # # #     text: str = Field(..., min_length=10, max_length=10000)
# # # #     language: Optional[str] = Field("en", max_length=10)

# # # # class ClearSessionRequest(BaseModel):
# # # #     session_id: Optional[str] = Field(None, max_length=100)

# # # # # API Endpoints
# # # # @app.post("/api/chatbot")
# # # # async def chatbot(request: ChatRequest):
# # # #     """Main chatbot endpoint with intelligent thinking."""
# # # #     try:
# # # #         session_id = request.session_id or str(uuid.uuid4())
# # # #         history = load_history(session_id)

# # # #         # Select appropriate agent
# # # #         query_lower = request.message.lower()
# # # #         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea', 'ache', 'hurt']):
# # # #             agent = symptom_analyzer_agent
# # # #         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction', 'side effect', 'ibuprofen', 'glutathion']):
# # # #             agent = drug_interaction_agent
# # # #         elif any(term in query_lower for term in ['what is', 'explain', 'define', 'meaning of']):
# # # #             agent = medical_term_agent
# # # #         elif any(term in query_lower for term in ['report', 'result', 'test', 'lab', 'x-ray', 'summary']):
# # # #             agent = report_analyzer_agent
# # # #         elif any(term in query_lower for term in ['creator', 'author', 'hadiqa', 'gohar', 'medicura about', 'who made']):
# # # #             agent = about_agent
# # # #         else:
# # # #             agent = general_health_agent

# # # #         result = await run_agent_with_thinking(agent, request.message, request.context)

# # # #         # Update chat history
# # # #         history.extend([
# # # #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# # # #             {"role": "assistant", "content": result, "timestamp": datetime.now().isoformat()}
# # # #         ])
# # # #         history = history[-20:]  # Keep last 20 messages
# # # #         save_history(session_id, history)

# # # #         return result

# # # #     except Exception as e:
# # # #         logger.error(f"Chatbot error: {str(e)}")
# # # #         return JSONResponse(
# # # #             status_code=500,
# # # #             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
# # # #         )

# # # # @app.post("/api/health/drug-interactions")
# # # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # # #     """Check drug interactions with thorough analysis."""
# # # #     try:
# # # #         if not input_data.medications or len(input_data.medications) == 0:
# # # #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# # # #         context = {
# # # #             "medications": input_data.medications,
# # # #             "age": input_data.age,
# # # #             "gender": input_data.gender,
# # # #             "existing_conditions": input_data.existing_conditions,
# # # #             "other_medications": input_data.other_medications
# # # #         }
        
# # # #         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
# # # #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Drug interaction error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/health/medical-term")
# # # # async def explain_medical_term(input_data: MedicalTermInput):
# # # #     """Explain medical terms with clarity."""
# # # #     try:
# # # #         if not input_data.term:
# # # #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# # # #         prompt = f"Explain the medical term: {input_data.term}"
# # # #         if input_data.language and input_data.language != "en":
# # # #             prompt += f" in {input_data.language} language"
        
# # # #         result = await run_agent_with_thinking(medical_term_agent, prompt)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Medical term error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/health/report-summarize")
# # # # async def summarize_medical_report(input_data: ReportTextInput):
# # # #     """Summarize medical reports with intelligent analysis."""
# # # #     try:
# # # #         if not input_data.text:
# # # #             raise HTTPException(status_code=400, detail="Report text is required")
        
# # # #         prompt = f"""
# # # #         Analyze and summarize this medical report:

# # # #         {input_data.text}

# # # #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# # # #         Focus on key findings, recommendations, and next steps.
# # # #         """

# # # #         result = await run_agent_with_thinking(report_analyzer_agent, prompt)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Report summary error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/chatbot/session/clear")
# # # # async def clear_session(request: ClearSessionRequest):
# # # #     """Clear chatbot session history."""
# # # #     try:
# # # #         session_id = request.session_id or "default_session"
# # # #         conn = get_db()
# # # #         with conn.cursor() as cur:
# # # #             cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
# # # #             conn.commit()
# # # #         conn.close()
# # # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # # #     except Exception as e:
# # # #         logger.error(f"Clear session error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # # @app.get("/health")
# # # # async def health_check():
# # # #     """Health check endpoint."""
# # # #     return {
# # # #         "status": "healthy",
# # # #         "timestamp": datetime.now().isoformat(),
# # # #         "version": "2.1.0",
# # # #         "agents_available": True,
# # # #         "thinking_enabled": True
# # # #     }

# # # # @app.get("/api/chatbot/sessions")
# # # # async def get_sessions():
# # # #     """Get active session count (for monitoring)."""
# # # #     try:
# # # #         conn = get_db()
# # # #         with conn.cursor() as cur:
# # # #             cur.execute("SELECT COUNT(*) FROM chat_sessions")
# # # #             active_sessions = cur.fetchone()[0]
# # # #             cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
# # # #             total_messages_result = cur.fetchone()[0]
# # # #             total_messages = total_messages_result if total_messages_result is not None else 0
# # # #         conn.close()
# # # #         return {
# # # #             "active_sessions": active_sessions,
# # # #             "total_messages": total_messages
# # # #         }
# # # #     except Exception as e:
# # # #         logger.error(f"Get sessions error: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Failed to retrieve session data")

# # # # if __name__ == "__main__":
# # # #     import uvicorn
# # # #     uvicorn.run(app, host="0.0.0.0", port=8000)







# # # # # import os
# # # # # import json
# # # # # import asyncio
# # # # # from datetime import datetime
# # # # # from fastapi import FastAPI, HTTPException
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from fastapi.responses import JSONResponse
# # # # # from pydantic import BaseModel, Field
# # # # # from dotenv import load_dotenv
# # # # # from typing import Optional, Dict, List, Any
# # # # # import logging
# # # # # from contextlib import asynccontextmanager
# # # # # import uuid
# # # # # import httpx
# # # # # import re
# # # # # import pymysql
# # # # # from sentence_transformers import SentenceTransformer

# # # # # # Import OpenAI Agents SDK components
# # # # # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # # # # # Import agent creation functions
# # # # # from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
# # # # # from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
# # # # # from medicura_agents.general_health_agent import create_general_health_agent
# # # # # from medicura_agents.medical_term_agent import create_medical_term_agent
# # # # # from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
# # # # # from medicura_agents.about_agent import create_about_agent

# # # # # # Configure logging
# # # # # logging.basicConfig(level=logging.INFO)
# # # # # logger = logging.getLogger(__name__)

# # # # # # Load environment variables
# # # # # load_dotenv()

# # # # # # TiDB Configuration
# # # # # DB_CONFIG = {
# # # # #     "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
# # # # #     "port": 4000,
# # # # #     "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
# # # # #     "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
# # # # #     "database": os.getenv("TIDB_DATABASE", "test"),
# # # # #     "charset": "utf8mb4",
# # # # #     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}
# # # # # }

# # # # # # Initialize embedding model
# # # # # embedder = SentenceTransformer('all-MiniLM-L6-v2')

# # # # # def generate_embedding(text: str) -> list:
# # # # #     """Convert text to a vector embedding."""
# # # # #     return embedder.encode(text).tolist()

# # # # # def save_embedding(session_id: str, user_id: str, content_type: str, text: str):
# # # # #     """Save text and its embedding to TiDB."""
# # # # #     embedding = generate_embedding(text)
# # # # #     embedding_str = ','.join(map(str, embedding))
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 INSERT INTO medical_embeddings (id, session_id, user_id, content_type, content_text, embedding)
# # # # #                 VALUES (%s, %s, %s, %s, %s, %s)
# # # # #             """, (str(uuid.uuid4()), session_id, user_id, content_type, text, embedding_str))
# # # # #             conn.commit()
# # # # #         conn.close()
# # # # #     except Exception as e:
# # # # #         logger.error(f"Failed to save embedding: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to save embedding")

# # # # # async def find_similar_queries(session_id: str, query: str):
# # # # #     """Find similar queries using vector search."""
# # # # #     embedding = generate_embedding(query)
# # # # #     embedding_str = ','.join(map(str, embedding))
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 SELECT content_text, VECTOR_COSINE_DISTANCE(embedding, %s) AS distance
# # # # #                 FROM medical_embeddings
# # # # #                 WHERE content_type = 'query' AND session_id != %s
# # # # #                 ORDER BY distance ASC
# # # # #                 LIMIT 5
# # # # #             """, (embedding_str, session_id))
# # # # #             results = cur.fetchall()
# # # # #         conn.close()
# # # # #         return [{"text": row[0], "distance": row[1]} for row in results]
# # # # #     except Exception as e:
# # # # #         logger.error(f"Vector search error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to perform vector search")

# # # # # async def search_medical_content(query: str):
# # # # #     """Search medical content using full-text search."""
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 SELECT content_text, MATCH(content_text) AGAINST(%s IN BOOLEAN MODE) AS score
# # # # #                 FROM medical_embeddings
# # # # #                 WHERE MATCH(content_text) AGAINST(%s IN BOOLEAN MODE)
# # # # #                 ORDER BY score DESC
# # # # #                 LIMIT 5
# # # # #             """, (query, query))
# # # # #             results = cur.fetchall()
# # # # #         conn.close()
# # # # #         return [{"text": row[0], "score": row[1]} for row in results]
# # # # #     except Exception as e:
# # # # #         logger.error(f"Full-text search error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to perform full-text search")

# # # # # async def fetch_pubmed_articles(query: str, max_results: int = 3):
# # # # #     """Fetch relevant medical articles from PubMed."""
# # # # #     async with httpx.AsyncClient() as client:
# # # # #         try:
# # # # #             response = await client.get(
# # # # #                 "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
# # # # #                 params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
# # # # #             )
# # # # #             response.raise_for_status()
# # # # #             data = response.json()
# # # # #             return [{"id": id} for id in data.get("esearchresult", {}).get("idlist", [])]
# # # # #         except Exception as e:
# # # # #             logger.error(f"PubMed API error: {str(e)}")
# # # # #             return []

# # # # # def get_db():
# # # # #     """Establish a connection to TiDB."""
# # # # #     try:
# # # # #         connection = pymysql.connect(**DB_CONFIG)
# # # # #         return connection
# # # # #     except pymysql.err.OperationalError as e:
# # # # #         logger.error(f"Failed to connect to TiDB: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# # # # # @asynccontextmanager
# # # # # async def lifespan(app: FastAPI):
# # # # #     """Manage application startup and shutdown."""
# # # # #     logger.info("Starting Medicura-AI Health Assistant")
# # # # #     logger.info("Connecting to TiDB...")
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 CREATE TABLE IF NOT EXISTS chat_sessions (
# # # # #                     session_id VARCHAR(100) PRIMARY KEY,
# # # # #                     history JSON NOT NULL,
# # # # #                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# # # # #                 )
# # # # #             """)
# # # # #             cur.execute("""
# # # # #                 CREATE TABLE IF NOT EXISTS medical_embeddings (
# # # # #                     id VARCHAR(100) PRIMARY KEY,
# # # # #                     session_id VARCHAR(100),
# # # # #                     user_id VARCHAR(100),
# # # # #                     content_type VARCHAR(50),
# # # # #                     content_text TEXT,
# # # # #                     embedding VECTOR(384),
# # # # #                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
# # # # #                     INDEX idx_session_id (session_id),
# # # # #                     INDEX idx_content_type (content_type),
# # # # #                     FULLTEXT (content_text)
# # # # #                 )
# # # # #             """)
# # # # #             conn.commit()
# # # # #         conn.close()
# # # # #         logger.info("TiDB connected and tables ready.")
# # # # #         yield
# # # # #     except Exception as e:
# # # # #         logger.error(f"Lifespan error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Application startup failed")
# # # # #     finally:
# # # # #         logger.info("Shutting down Medicura-AI Health Assistant")

# # # # # app = FastAPI(
# # # # #     title="Medicura-AI Health Assistant",
# # # # #     description="AI-powered health assistant for symptom analysis and medical queries",
# # # # #     version="2.1.0",
# # # # #     lifespan=lifespan,
# # # # #     docs_url="/api/docs",
# # # # #     redoc_url="/api/redoc"
# # # # # )

# # # # # # CORS Configuration
# # # # # app.add_middleware(
# # # # #     CORSMiddleware,
# # # # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
# # # # #     allow_credentials=True,
# # # # #     allow_methods=["*"],
# # # # #     allow_headers=["*"],
# # # # # )

# # # # # # Environment Variable Validation
# # # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # # if not GEMINI_API_KEY:
# # # # #     logger.error("GEMINI_API_KEY not found in environment variables")
# # # # #     raise ValueError("GEMINI_API_KEY environment variable is required")

# # # # # TIDB_HOST = os.getenv("TIDB_HOST")
# # # # # TIDB_USERNAME = os.getenv("TIDB_USERNAME")
# # # # # TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
# # # # # TIDB_DATABASE = os.getenv("TIDB_DATABASE")

# # # # # # AI Agent Initialization
# # # # # external_client = AsyncOpenAI(
# # # # #     api_key=GEMINI_API_KEY,
# # # # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # # #     http_client=httpx.AsyncClient(timeout=60.0)
# # # # # )

# # # # # model = OpenAIChatCompletionsModel(
# # # # #     model="gemini-2.0-flash",
# # # # #     openai_client=external_client,
# # # # # )

# # # # # model_settings = ModelSettings(
# # # # #     temperature=0.7,
# # # # #     top_p=0.9,
# # # # #     max_tokens=2048,
# # # # # )

# # # # # config = RunConfig(
# # # # #     model=model,
# # # # #     model_provider=external_client,
# # # # #     model_settings=model_settings,
# # # # #     tracing_disabled=True,
# # # # # )

# # # # # # Initialize Agents
# # # # # symptom_analyzer_agent = create_symptom_analyzer_agent(model)
# # # # # drug_interaction_agent = create_drug_interaction_agent(model)
# # # # # general_health_agent = create_general_health_agent(model)
# # # # # medical_term_agent = create_medical_term_agent(model)
# # # # # report_analyzer_agent = create_report_analyzer_agent(model)
# # # # # about_agent = create_about_agent(model)

# # # # # def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
# # # # #     """Extract JSON from agent response with multiple fallback methods."""
# # # # #     try:
# # # # #         try:
# # # # #             return json.loads(response.strip())
# # # # #         except json.JSONDecodeError:
# # # # #             pass
# # # # #         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
# # # # #         if json_match:
# # # # #             return json.loads(json_match.group(1))
# # # # #         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
# # # # #         if brace_match:
# # # # #             return json.loads(brace_match.group(0))
# # # # #         return {
# # # # #             "summary": response,
# # # # #             "detailed_analysis": "Detailed analysis based on your query",
# # # # #             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
# # # # #             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
# # # # #             "type": "general"
# # # # #         }
# # # # #     except Exception as e:
# # # # #         logger.warning(f"JSON extraction failed: {str(e)}")
# # # # #         return None

# # # # # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# # # # #     """Run agent with enhanced thinking and robust error handling."""
# # # # #     try:
# # # # #         thinking_prompt = f"""
# # # # #         USER QUERY: {prompt}
# # # # #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
        
# # # # #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# # # # #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# # # # #         """
        
# # # # #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# # # # #         parsed_response = extract_json_from_response(result.final_output)
        
# # # # #         if parsed_response:
# # # # #             parsed_response.update({
# # # # #                 "timestamp": datetime.now().isoformat(),
# # # # #                 "success": True,
# # # # #                 "thinking_applied": True
# # # # #             })
# # # # #             return parsed_response
# # # # #         else:
# # # # #             return create_intelligent_response(result.final_output, prompt)
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Agent error: {str(e)}")
# # # # #         return create_intelligent_response(f"Analysis of: {prompt}")

# # # # # def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
# # # # #     """Create a well-structured response from text."""
# # # # #     return {
# # # # #         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
# # # # #         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
# # # # #         "recommendations": [
# # # # #             "Consult with a healthcare provider",
# # # # #             "Provide complete medical history for assessment",
# # # # #             "Follow evidence-based medical guidance"
# # # # #         ],
# # # # #         "when_to_seek_help": [
# # # # #             "Immediately for severe or emergency symptoms",
# # # # #             "Within 24-48 hours for persistent concerns",
# # # # #             "Routinely for preventive care"
# # # # #         ],
# # # # #         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
# # # # #         "type": "general",
# # # # #         "timestamp": datetime.now().isoformat(),
# # # # #         "success": True,
# # # # #         "thinking_applied": True
# # # # #     }

# # # # # def load_history(session_id: str) -> List[dict]:
# # # # #     """Load chat history from TiDB."""
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
# # # # #             result = cur.fetchone()
# # # # #         conn.close()
# # # # #         return json.loads(result[0]) if result else []
# # # # #     except Exception as e:
# # # # #         logger.error(f"Failed to load history: {str(e)}")
# # # # #         return []

# # # # # def save_history(session_id: str, history: List[dict]):
# # # # #     """Save chat history to TiDB."""
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 INSERT INTO chat_sessions (session_id, history)
# # # # #                 VALUES (%s, %s)
# # # # #                 ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
# # # # #             """, (session_id, json.dumps(history), json.dumps(history)))
# # # # #             conn.commit()
# # # # #         conn.close()
# # # # #     except Exception as e:
# # # # #         logger.error(f"Failed to save history: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to save chat history")

# # # # # # Pydantic Models
# # # # # class ChatRequest(BaseModel):
# # # # #     message: str = Field(..., min_length=1, max_length=1000)
# # # # #     session_id: Optional[str] = Field(None, max_length=100)
# # # # #     context: Optional[dict] = None

# # # # # class DrugInteractionInput(BaseModel):
# # # # #     medications: List[str] = Field(..., min_items=1, max_items=10)
# # # # #     age: Optional[int] = Field(None, ge=0, le=120)
# # # # #     gender: Optional[str] = Field(None, max_length=20)
# # # # #     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
# # # # #     other_medications: Optional[List[str]] = Field(None, max_items=20)

# # # # # class MedicalTermInput(BaseModel):
# # # # #     term: str = Field(..., min_length=1, max_length=100)
# # # # #     language: Optional[str] = Field("en", max_length=10)

# # # # # class ReportTextInput(BaseModel):
# # # # #     text: str = Field(..., min_length=10, max_length=10000)
# # # # #     language: Optional[str] = Field("en", max_length=10)

# # # # # class ClearSessionRequest(BaseModel):
# # # # #     session_id: Optional[str] = Field(None, max_length=100)

# # # # # # API Endpoints
# # # # # @app.post("/api/chatbot")
# # # # # async def chatbot(request: ChatRequest):
# # # # #     """Main chatbot endpoint with intelligent thinking and vector search."""
# # # # #     try:
# # # # #         session_id = request.session_id or str(uuid.uuid4())
# # # # #         user_id = str(uuid.uuid4())
        
# # # # #         # Step 1: Ingest & Index Data
# # # # #         save_embedding(session_id, user_id, "query", request.message)
        
# # # # #         # Step 2: Search Your Data
# # # # #         similar_queries = await find_similar_queries(session_id, request.message)
# # # # #         related_content = await search_medical_content(request.message)
        
# # # # #         # Step 3: Invoke External Tools
# # # # #         pubmed_articles = await fetch_pubmed_articles(request.message)
        
# # # # #         # Step 4: Select appropriate agent
# # # # #         context = {
# # # # #             "similar_queries": similar_queries,
# # # # #             "related_content": related_content,
# # # # #             "pubmed_articles": pubmed_articles,
# # # # #             **(request.context or {})
# # # # #         }
# # # # #         query_lower = request.message.lower()
# # # # #         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea', 'ache', 'hurt']):
# # # # #             agent = symptom_analyzer_agent
# # # # #         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction', 'side effect', 'ibuprofen', 'glutathion']):
# # # # #             agent = drug_interaction_agent
# # # # #         elif any(term in query_lower for term in ['what is', 'explain', 'define', 'meaning of']):
# # # # #             agent = medical_term_agent
# # # # #         elif any(term in query_lower for term in ['report', 'result', 'test', 'lab', 'x-ray', 'summary']):
# # # # #             agent = report_analyzer_agent
# # # # #         elif any(term in query_lower for term in ['creator', 'author', 'hadiqa', 'gohar', 'medicura about', 'who made']):
# # # # #             agent = about_agent
# # # # #         else:
# # # # #             agent = general_health_agent
        
# # # # #         # Step 5: Run agent with enhanced context
# # # # #         result = await run_agent_with_thinking(agent, request.message, context)
        
# # # # #         # Step 6: Update chat history
# # # # #         history = load_history(session_id)
# # # # #         history.extend([
# # # # #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# # # # #             {"role": "assistant", "content": result, "timestamp": datetime.now().isoformat()}
# # # # #         ])
# # # # #         history = history[-20:]
# # # # #         save_history(session_id, history)
        
# # # # #         return result
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Chatbot error: {str(e)}")
# # # # #         return JSONResponse(
# # # # #             status_code=500,
# # # # #             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
# # # # #         )

# # # # # @app.post("/api/health/drug-interactions")
# # # # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # # # #     """Check drug interactions with thorough analysis."""
# # # # #     try:
# # # # #         if not input_data.medications or len(input_data.medications) == 0:
# # # # #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# # # # #         session_id = str(uuid.uuid4())
# # # # #         user_id = str(uuid.uuid4())
        
# # # # #         # Save query embedding
# # # # #         query = f"Check interactions for: {', '.join(input_data.medications)}"
# # # # #         save_embedding(session_id, user_id, "query", query)
        
# # # # #         context = {
# # # # #             "medications": input_data.medications,
# # # # #             "age": input_data.age,
# # # # #             "gender": input_data.gender,
# # # # #             "existing_conditions": input_data.existing_conditions,
# # # # #             "other_medications": input_data.other_medications
# # # # #         }
        
# # # # #         prompt = query
# # # # #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# # # # #         return result
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Drug interaction error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/health/medical-term")
# # # # # async def explain_medical_term(input_data: MedicalTermInput):
# # # # #     """Explain medical terms with clarity."""
# # # # #     try:
# # # # #         if not input_data.term:
# # # # #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# # # # #         session_id = str(uuid.uuid4())
# # # # #         user_id = str(uuid.uuid4())
        
# # # # #         # Save term embedding
# # # # #         save_embedding(session_id, user_id, "term", input_data.term)
        
# # # # #         # Search for related content
# # # # #         related_content = await search_medical_content(input_data.term)
# # # # #         context = {"related_content": related_content}
        
# # # # #         prompt = f"Explain the medical term: {input_data.term}"
# # # # #         if input_data.language and input_data.language != "en":
# # # # #             prompt += f" in {input_data.language} language"
        
# # # # #         result = await run_agent_with_thinking(medical_term_agent, prompt, context)
# # # # #         return result
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Medical term error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/health/report-summarize")
# # # # # async def summarize_medical_report(input_data: ReportTextInput):
# # # # #     """Summarize medical reports with intelligent analysis."""
# # # # #     try:
# # # # #         if not input_data.text:
# # # # #             raise HTTPException(status_code=400, detail="Report text is required")
        
# # # # #         session_id = str(uuid.uuid4())
# # # # #         user_id = str(uuid.uuid4())
        
# # # # #         # Save report embedding
# # # # #         save_embedding(session_id, user_id, "report", input_data.text)
        
# # # # #         # Find similar reports
# # # # #         similar_reports = await find_similar_queries(session_id, input_data.text)
# # # # #         context = {"similar_reports": similar_reports}
        
# # # # #         prompt = f"""
# # # # #         Analyze and summarize this medical report:
# # # # #         {input_data.text}
# # # # #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# # # # #         Focus on key findings, recommendations, and next steps.
# # # # #         """
        
# # # # #         result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
# # # # #         return result
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Report summary error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/chatbot/session/clear")
# # # # # async def clear_session(request: ClearSessionRequest):
# # # # #     """Clear chatbot session history."""
# # # # #     try:
# # # # #         session_id = request.session_id or "default_session"
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
# # # # #             conn.commit()
# # # # #         conn.close()
# # # # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # # # #     except Exception as e:
# # # # #         logger.error(f"Clear session error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # # # @app.get("/health")
# # # # # async def health_check():
# # # # #     """Health check endpoint."""
# # # # #     return {
# # # # #         "status": "healthy",
# # # # #         "timestamp": datetime.now().isoformat(),
# # # # #         "version": "2.1.0",
# # # # #         "agents_available": True,
# # # # #         "thinking_enabled": True
# # # # #     }

# # # # # @app.get("/api/chatbot/sessions")
# # # # # async def get_sessions():
# # # # #     """Get active session count (for monitoring)."""
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("SELECT COUNT(*) FROM chat_sessions")
# # # # #             active_sessions = cur.fetchone()[0]
# # # # #             cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
# # # # #             total_messages_result = cur.fetchone()[0]
# # # # #             total_messages = total_messages_result if total_messages_result is not None else 0
# # # # #         conn.close()
# # # # #         return {
# # # # #             "active_sessions": active_sessions,
# # # # #             "total_messages": total_messages
# # # # #         }
# # # # #     except Exception as e:
# # # # #         logger.error(f"Get sessions error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to retrieve session data")

# # # # # @app.get("/api/health/analytics")
# # # # # async def get_health_analytics():
# # # # #     """Get analytics on common symptoms."""
# # # # #     try:
# # # # #         conn = get_db()
# # # # #         with conn.cursor() as cur:
# # # # #             cur.execute("""
# # # # #                 SELECT JSON_EXTRACT(history, '$[*].content') AS messages
# # # # #                 FROM chat_sessions
# # # # #             """)
# # # # #             results = cur.fetchall()
# # # # #             symptom_counts = {}
# # # # #             for row in results:
# # # # #                 messages = json.loads(row[0]) if row[0] else []
# # # # #                 for msg in messages:
# # # # #                     if msg.get("role") == "user":
# # # # #                         text = msg.get("content", "").lower()
# # # # #                         if any(term in text for term in ['headache', 'fever', 'pain', 'cough']):
# # # # #                             for term in ['headache', 'fever', 'pain', 'cough']:
# # # # #                                 if term in text:
# # # # #                                     symptom_counts[term] = symptom_counts.get(term, 0) + 1
# # # # #         conn.close()
# # # # #         return {"top_symptoms": [{"symptom": k, "count": v} for k, v in sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]]}
# # # # #     except Exception as e:
# # # # #         logger.error(f"Analytics error: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

# # # # # if __name__ == "__main__":
# # # # #     import uvicorn
# # # # #     uvicorn.run(app, host="0.0.0.0", port=8000)



# # # import os
# # # import json
# # # import asyncio
# # # from datetime import datetime
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from fastapi.responses import JSONResponse
# # # from pydantic import BaseModel, Field
# # # from dotenv import load_dotenv
# # # from typing import Optional, Dict, List, Any
# # # import logging
# # # from contextlib import asynccontextmanager
# # # import uuid
# # # import httpx
# # # import re
# # # import pymysql

# # # # Import OpenAI Agents SDK components (assumed to be available)
# # # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # # # Import agent creation functions from medicura_agents and specialist_agents
# # # from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
# # # from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
# # # from medicura_agents.general_health_agent import create_general_health_agent
# # # from medicura_agents.medical_term_agent import create_medical_term_agent
# # # from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
# # # from medicura_agents.about_agent import create_about_agent
# # # from specialist_agents.cardiology_ai import create_cardiology_agent
# # # from specialist_agents.dermatology_ai import create_dermatology_agent
# # # from specialist_agents.neurology_ai import create_neurology_agent
# # # from specialist_agents.pulmonology_ai import create_pulmonology_agent
# # # from specialist_agents.ophthalmology_ai import create_ophthalmology_agent
# # # from specialist_agents.dental_ai import create_dental_agent
# # # from specialist_agents.allergy_immunology_ai import create_allergy_immunology_agent
# # # from specialist_agents.pediatrics_ai import create_pediatrics_agent
# # # from specialist_agents.orthopedics_ai import create_orthopedics_agent
# # # from specialist_agents.mental_health_ai import create_mental_health_agent
# # # from specialist_agents.endocrinology_ai import create_endocrinology_agent
# # # from specialist_agents.gastroenterology_ai import create_gastroenterology_agent
# # # from specialist_agents.radiology_ai import create_radiology_agent
# # # from specialist_agents.infectious_disease_ai import create_infectious_disease_agent
# # # from specialist_agents.vaccination_advisor_ai import create_vaccination_advisor_agent

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # # Load environment variables from .env file
# # # load_dotenv()

# # # # TiDB Configuration with minimal SSL
# # # DB_CONFIG = {
# # #     "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
# # #     "port": 4000,
# # #     "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
# # #     "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
# # #     "database": os.getenv("TIDB_DATABASE", "test"),
# # #     "charset": "utf8mb4",
# # #     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
# # # }

# # # def get_db():
# # #     """Establish a connection to TiDB."""
# # #     try:
# # #         connection = pymysql.connect(**DB_CONFIG)
# # #         return connection
# # #     except pymysql.err.OperationalError as e:
# # #         logger.error(f"Failed to connect to TiDB: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# # # @asynccontextmanager
# # # async def lifespan(app: FastAPI):
# # #     """Manage application startup and shutdown."""
# # #     logger.info("Starting Medicura-AI Health Assistant")
# # #     logger.info("Connecting to TiDB...")
# # #     try:
# # #         conn = get_db()
# # #         with conn.cursor() as cur:
# # #             cur.execute("""
# # #                 CREATE TABLE IF NOT EXISTS chat_sessions (
# # #                     session_id VARCHAR(100) PRIMARY KEY,
# # #                     history JSON NOT NULL,
# # #                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# # #                 )
# # #             """)
# # #             cur.execute("""
# # #                 CREATE TABLE IF NOT EXISTS specialist_vectors (
# # #                     id VARCHAR(100) PRIMARY KEY,
# # #                     specialty VARCHAR(50) NOT NULL,
# # #                     content TEXT NOT NULL,
# # #                     embedding JSON NOT NULL,
# # #                     metadata JSON,
# # #                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# # #                 )
# # #             """)
# # #             conn.commit()
# # #         conn.close()
# # #         logger.info("TiDB connected and tables ready.")
# # #         yield
# # #     except Exception as e:
# # #         logger.error(f"Lifespan error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Application startup failed")
# # #     finally:
# # #         logger.info("Shutting down Medicura-AI Health Assistant")

# # # app = FastAPI(
# # #     title="Medicura-AI Health Assistant",
# # #     description="AI-powered health assistant for symptom analysis and medical queries",
# # #     version="2.1.0",
# # #     lifespan=lifespan,
# # #     docs_url="/api/docs",
# # #     redoc_url="/api/redoc"
# # # )

# # # # CORS Configuration
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # Environment Variable Validation
# # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # if not GEMINI_API_KEY:
# # #     logger.error("GEMINI_API_KEY not found in environment variables")
# # #     raise ValueError("GEMINI_API_KEY environment variable is required")

# # # # AI Agent Initialization
# # # external_client = AsyncOpenAI(
# # #     api_key=GEMINI_API_KEY,
# # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # #     http_client=httpx.AsyncClient(timeout=60.0)
# # # )

# # # model = OpenAIChatCompletionsModel(
# # #     model="gemini-2.0-flash",
# # #     openai_client=external_client,
# # # )

# # # model_settings = ModelSettings(
# # #     temperature=0.7,
# # #     top_p=0.9,
# # #     max_tokens=2048,
# # # )

# # # config = RunConfig(
# # #     model=model,
# # #     model_provider=external_client,
# # #     model_settings=model_settings,
# # #     tracing_disabled=True,
# # # )

# # # # Initialize All Agents
# # # symptom_analyzer_agent = create_symptom_analyzer_agent(model)
# # # drug_interaction_agent = create_drug_interaction_agent(model)
# # # general_health_agent = create_general_health_agent(model)
# # # medical_term_agent = create_medical_term_agent(model)
# # # report_analyzer_agent = create_report_analyzer_agent(model)
# # # about_agent = create_about_agent(model)
# # # cardiology_agent = create_cardiology_agent(model)
# # # dermatology_agent = create_dermatology_agent(model)
# # # neurology_agent = create_neurology_agent(model)
# # # pulmonology_agent = create_pulmonology_agent(model)
# # # ophthalmology_agent = create_ophthalmology_agent(model)
# # # dental_agent = create_dental_agent(model)
# # # allergy_immunology_agent = create_allergy_immunology_agent(model)
# # # pediatrics_agent = create_pediatrics_agent(model)
# # # orthopedics_agent = create_orthopedics_agent(model)
# # # mental_health_agent = create_mental_health_agent(model)
# # # endocrinology_agent = create_endocrinology_agent(model)
# # # gastroenterology_agent = create_gastroenterology_agent(model)
# # # radiology_agent = create_radiology_agent(model)
# # # infectious_disease_agent = create_infectious_disease_agent(model)
# # # vaccination_advisor_agent = create_vaccination_advisor_agent(model)

# # # def generate_embedding(text: str) -> List[float]:
# # #     """Generate embedding using Gemini API (simplified placeholder)."""
# # #     # Replace with actual embedding generation logic using Gemini API
# # #     # This is a mock implementation; integrate with your embedding model
# # #     return [0.1] * 768  # Mock 768-dimensional embedding

# # # def ingest_specialty_data():
# # #     """Ingest sample data into specialist_vectors table."""
# # #     sample_data = [
# # #         {"id": "cardio_1", "specialty": "cardiology", "content": "chest pain, shortness of breath", "metadata": {"diagnosis": "Possible angina"}},
# # #         {"id": "cardio_2", "specialty": "cardiology", "content": "palpitations, fatigue", "metadata": {"diagnosis": "Possible arrhythmia"}},
# # #         {"id": "derm_1", "specialty": "dermatology", "content": "red itchy rash on arm", "metadata": {"diagnosis": "Possible eczema"}},
# # #         {"id": "derm_2", "specialty": "dermatology", "content": "dry skin patches", "metadata": {"diagnosis": "Possible psoriasis"}},
# # #         {"id": "neuro_1", "specialty": "neurology", "content": "headache, dizziness", "metadata": {"diagnosis": "Possible migraine"}},
# # #         {"id": "neuro_2", "specialty": "neurology", "content": "numbness in hands", "metadata": {"diagnosis": "Possible neuropathy"}},
# # #         {"id": "pulmo_1", "specialty": "pulmonology", "content": "persistent cough", "metadata": {"diagnosis": "Possible bronchitis"}},
# # #         {"id": "pulmo_2", "specialty": "pulmonology", "content": "wheezing, shortness of breath", "metadata": {"diagnosis": "Possible asthma"}},
# # #         {"id": "ophtha_1", "specialty": "ophthalmology", "content": "blurry vision", "metadata": {"diagnosis": "Possible cataract"}},
# # #         {"id": "dental_1", "specialty": "dental", "content": "toothache, swelling", "metadata": {"diagnosis": "Possible abscess"}},
# # #         {"id": "allergy_1", "specialty": "allergy_immunology", "content": "sneezing, runny nose", "metadata": {"diagnosis": "Possible allergies"}},
# # #         {"id": "peds_1", "specialty": "pediatrics", "content": "fever in child", "metadata": {"diagnosis": "Possible infection"}},
# # #         {"id": "ortho_1", "specialty": "orthopedics", "content": "joint pain", "metadata": {"diagnosis": "Possible arthritis"}},
# # #         {"id": "mental_1", "specialty": "mental_health", "content": "anxiety, stress", "metadata": {"diagnosis": "Possible anxiety disorder"}},
# # #         {"id": "endo_1", "specialty": "endocrinology", "content": "fatigue, weight gain", "metadata": {"diagnosis": "Possible hypothyroidism"}},
# # #         {"id": "gastro_1", "specialty": "gastroenterology", "content": "stomach pain", "metadata": {"diagnosis": "Possible gastritis"}},
# # #         {"id": "radio_1", "specialty": "radiology", "content": "abnormal x-ray", "metadata": {"diagnosis": "Possible fracture"}},
# # #         {"id": "infect_1", "specialty": "infectious_disease", "content": "fever, chills", "metadata": {"diagnosis": "Possible flu"}},
# # #         {"id": "vacc_1", "specialty": "vaccination_advisor", "content": "vaccine schedule", "metadata": {"recommendation": "Consult pediatrician"}}
# # #     ]
# # #     conn = get_db()
# # #     try:
# # #         with conn.cursor() as cur:
# # #             for item in sample_data:
# # #                 embedding = generate_embedding(item["content"])
# # #                 cur.execute("""
# # #                     INSERT INTO specialist_vectors (id, specialty, content, embedding, metadata)
# # #                     VALUES (%s, %s, %s, %s, %s)
# # #                     ON DUPLICATE KEY UPDATE content = VALUES(content), embedding = VALUES(embedding), metadata = VALUES(metadata)
# # #                 """, (item["id"], item["specialty"], item["content"], json.dumps(embedding), json.dumps(item["metadata"])))
# # #             conn.commit()
# # #     finally:
# # #         conn.close()

# # # def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# # #     """Search similar cases in specialist_vectors using cosine similarity."""
# # #     embedding = generate_embedding(query)
# # #     conn = get_db()
# # #     try:
# # #         with conn.cursor() as cur:
# # #             # TiDB vector search syntax (simplified; adjust based on actual capabilities)
# # #             cur.execute("""
# # #                 SELECT id, content, metadata
# # #                 FROM specialist_vectors
# # #                 WHERE specialty = %s
# # #                 ORDER BY (
# # #                     1 - (
# # #                         (SELECT SUM(a * b) FROM JSON_TABLE(
# # #                             CAST(JSON_EXTRACT(embedding, '$') AS JSON),
# # #                             '$[*]' COLUMNS (a DOUBLE PATH '$')
# # #                         ) jt1,
# # #                         JSON_TABLE(
# # #                             %s,
# # #                             '$[*]' COLUMNS (b DOUBLE PATH '$')
# # #                         ) jt2) /
# # #                         (SQRT((SELECT SUM(a * a) FROM JSON_TABLE(
# # #                             CAST(JSON_EXTRACT(embedding, '$') AS JSON),
# # #                             '$[*]' COLUMNS (a DOUBLE PATH '$')
# # #                         ))) * SQRT((SELECT SUM(b * b) FROM JSON_TABLE(
# # #                             %s,
# # #                             '$[*]' COLUMNS (b DOUBLE PATH '$')
# # #                         ))))
# # #                     )
# # #                 ) LIMIT %s
# # #             """, (specialty, json.dumps(embedding), json.dumps(embedding), top_k))
# # #             results = cur.fetchall()
# # #             return [{"id": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]
# # #     except Exception as e:
# # #         logger.error(f"Vector search error: {str(e)}")
# # #         return []
# # #     finally:
# # #         conn.close()

# # # def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
# # #     """Extract JSON from agent response with multiple fallback methods."""
# # #     try:
# # #         try:
# # #             return json.loads(response.strip())
# # #         except json.JSONDecodeError:
# # #             pass
# # #         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
# # #         if json_match:
# # #             return json.loads(json_match.group(1))
# # #         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
# # #         if brace_match:
# # #             return json.loads(brace_match.group(0))
# # #         return {
# # #             "summary": response,
# # #             "detailed_analysis": "Detailed analysis based on your query",
# # #             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
# # #             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
# # #             "type": "general"
# # #         }
# # #     except Exception as e:
# # #         logger.warning(f"JSON extraction failed: {str(e)}")
# # #         return None

# # # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# # #     """Run agent with enhanced thinking and robust error handling."""
# # #     try:
# # #         # Determine specialty from agent name or context
# # #         specialty = "general"
# # #         if hasattr(agent, 'specialty'):
# # #             specialty = agent.specialty
# # #         elif context and "specialty" in context:
# # #             specialty = context["specialty"]

# # #         thinking_prompt = f"""
# # #         USER QUERY: {prompt}
# # #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
# # #         SIMILAR CASES: {json.dumps(search_similar_cases(prompt, specialty))}
        
# # #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# # #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# # #         """
        
# # #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# # #         parsed_response = extract_json_from_response(result.final_output)
        
# # #         if parsed_response:
# # #             parsed_response.update({
# # #                 "timestamp": datetime.now().isoformat(),
# # #                 "success": True,
# # #                 "thinking_applied": True
# # #             })
# # #             return parsed_response
# # #         else:
# # #             return create_intelligent_response(result.final_output, prompt)
        
# # #     except Exception as e:
# # #         logger.error(f"Agent error: {str(e)}")
# # #         return create_intelligent_response(f"Analysis of: {prompt}")

# # # def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
# # #     """Create a well-structured response from text."""
# # #     return {
# # #         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
# # #         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
# # #         "recommendations": [
# # #             "Consult with a healthcare provider",
# # #             "Provide complete medical history for assessment",
# # #             "Follow evidence-based medical guidance"
# # #         ],
# # #         "when_to_seek_help": [
# # #             "Immediately for severe or emergency symptoms",
# # #             "Within 24-48 hours for persistent concerns",
# # #             "Routinely for preventive care"
# # #         ],
# # #         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
# # #         "type": "general",
# # #         "timestamp": datetime.now().isoformat(),
# # #         "success": True,
# # #         "thinking_applied": True
# # #     }

# # # def load_history(session_id: str) -> List[dict]:
# # #     """Load chat history from TiDB."""
# # #     try:
# # #         conn = get_db()
# # #         with conn.cursor() as cur:
# # #             cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
# # #             result = cur.fetchone()
# # #         conn.close()
# # #         return json.loads(result[0]) if result else []
# # #     except Exception as e:
# # #         logger.error(f"Failed to load history: {str(e)}")
# # #         return []

# # # def save_history(session_id: str, history: List[dict]):
# # #     """Save chat history to TiDB."""
# # #     try:
# # #         conn = get_db()
# # #         with conn.cursor() as cur:
# # #             cur.execute("""
# # #                 INSERT INTO chat_sessions (session_id, history)
# # #                 VALUES (%s, %s)
# # #                 ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
# # #             """, (session_id, json.dumps(history), json.dumps(history)))
# # #             conn.commit()
# # #         conn.close()
# # #     except Exception as e:
# # #         logger.error(f"Failed to save history: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Failed to save chat history")

# # # # Pydantic Models
# # # class ChatRequest(BaseModel):
# # #     message: str = Field(..., min_length=1, max_length=1000)
# # #     session_id: Optional[str] = Field(None, max_length=100)
# # #     context: Optional[dict] = None

# # # class DrugInteractionInput(BaseModel):
# # #     medications: List[str] = Field(..., min_items=1, max_items=10)
# # #     age: Optional[int] = Field(None, ge=0, le=120)
# # #     gender: Optional[str] = Field(None, max_length=20)
# # #     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
# # #     other_medications: Optional[List[str]] = Field(None, max_items=20)

# # # class MedicalTermInput(BaseModel):
# # #     term: str = Field(..., min_length=1, max_length=100)
# # #     language: Optional[str] = Field("en", max_length=10)

# # # class ReportTextInput(BaseModel):
# # #     text: str = Field(..., min_length=10, max_length=10000)
# # #     language: Optional[str] = Field("en", max_length=10)

# # # class ClearSessionRequest(BaseModel):
# # #     session_id: Optional[str] = Field(None, max_length=100)

# # # # API Endpoints
# # # @app.post("/api/chatbot")
# # # async def chatbot(request: ChatRequest):
# # #     """Main chatbot endpoint with intelligent thinking and specialty support."""
# # #     try:
# # #         session_id = request.session_id or str(uuid.uuid4())
# # #         history = load_history(session_id)

# # #         # Ingest specialty data on first run
# # #         if not history:
# # #             ingest_specialty_data()

# # #         # Select appropriate agent based on specialty and keywords
# # #         query_lower = request.message.lower()
# # #         specialty_map = {
# # #             "cardiology": ["chest pain", "heart", "palpitations", "shortness of breath"],
# # #             "dermatology": ["rash", "skin", "itch", "eczema"],
# # #             "neurology": ["headache", "seizure", "numbness", "dizziness"],
# # #             "pulmonology": ["cough", "breathing", "asthma", "lung"],
# # #             "ophthalmology": ["vision", "eye", "blurry", "glaucoma"],
# # #             "dental": ["tooth", "gum", "oral", "cavity"],
# # #             "allergy_immunology": ["allergy", "immune", "sneezing", "anaphylaxis"],
# # #             "pediatrics": ["child", "fever", "growth", "pediatric"],
# # #             "orthopedics": ["bone", "joint", "fracture", "arthritis"],
# # #             "mental_health": ["anxiety", "depression", "stress", "mental"],
# # #             "endocrinology": ["diabetes", "thyroid", "hormone", "metabolism"],
# # #             "gastroenterology": ["stomach", "digestion", "ulcer", "gastro"],
# # #             "radiology": ["x-ray", "imaging", "scan", "radiology"],
# # #             "infectious_disease": ["infection", "virus", "bacteria", "fever"],
# # #             "vaccination_advisor": ["vaccine", "immunization", "shot", "vaccination"],
# # #             "symptom": ["symptom", "pain", "fever", "headache", "nausea", "ache", "hurt"],
# # #             "drug": ["drug", "medication", "pill", "dose", "interaction", "side effect", "ibuprofen", "glutathion"],
# # #             "medical_term": ["what is", "explain", "define", "meaning of"],
# # #             "report": ["report", "result", "test", "lab", "x-ray", "summary"],
# # #             "about": ["creator", "author", "hadiqa", "gohar", "medicura about", "who made"]
# # #         }

# # #         selected_specialty = "general"
# # #         selected_agent = general_health_agent
# # #         for specialty, keywords in specialty_map.items():
# # #             if any(keyword in query_lower for keyword in keywords):
# # #                 selected_specialty = specialty
# # #                 if specialty == "cardiology":
# # #                     selected_agent = cardiology_agent
# # #                 elif specialty == "dermatology":
# # #                     selected_agent = dermatology_agent
# # #                 elif specialty == "neurology":
# # #                     selected_agent = neurology_agent
# # #                 elif specialty == "pulmonology":
# # #                     selected_agent = pulmonology_agent
# # #                 elif specialty == "ophthalmology":
# # #                     selected_agent = ophthalmology_agent
# # #                 elif specialty == "dental":
# # #                     selected_agent = dental_agent
# # #                 elif specialty == "allergy_immunology":
# # #                     selected_agent = allergy_immunology_agent
# # #                 elif specialty == "pediatrics":
# # #                     selected_agent = pediatrics_agent
# # #                 elif specialty == "orthopedics":
# # #                     selected_agent = orthopedics_agent
# # #                 elif specialty == "mental_health":
# # #                     selected_agent = mental_health_agent
# # #                 elif specialty == "endocrinology":
# # #                     selected_agent = endocrinology_agent
# # #                 elif specialty == "gastroenterology":
# # #                     selected_agent = gastroenterology_agent
# # #                 elif specialty == "radiology":
# # #                     selected_agent = radiology_agent
# # #                 elif specialty == "infectious_disease":
# # #                     selected_agent = infectious_disease_agent
# # #                 elif specialty == "vaccination_advisor":
# # #                     selected_agent = vaccination_advisor_agent
# # #                 elif specialty == "symptom":
# # #                     selected_agent = symptom_analyzer_agent
# # #                 elif specialty == "drug":
# # #                     selected_agent = drug_interaction_agent
# # #                 elif specialty == "medical_term":
# # #                     selected_agent = medical_term_agent
# # #                 elif specialty == "report":
# # #                     selected_agent = report_analyzer_agent
# # #                 elif specialty == "about":
# # #                     selected_agent = about_agent
# # #                 break

# # #         # Run agent with thinking mode and vector search context
# # #         context = request.context or {}
# # #         context["specialty"] = selected_specialty
# # #         context["similar_cases"] = search_similar_cases(request.message, selected_specialty)
# # #         result = await run_agent_with_thinking(selected_agent, request.message, context)

# # #         # Update chat history
# # #         history.extend([
# # #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# # #             {"role": "assistant", "content": json.dumps(result), "timestamp": datetime.now().isoformat()}
# # #         ])
# # #         history = history[-20:]  # Keep last 20 messages
# # #         save_history(session_id, history)

# # #         return JSONResponse(content=result)

# # #     except Exception as e:
# # #         logger.error(f"Chatbot error: {str(e)}")
# # #         return JSONResponse(
# # #             status_code=500,
# # #             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
# # #         )

# # # @app.post("/api/health/drug-interactions")
# # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # #     """Check drug interactions with thorough analysis."""
# # #     try:
# # #         if not input_data.medications or len(input_data.medications) == 0:
# # #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# # #         context = {
# # #             "medications": input_data.medications,
# # #             "age": input_data.age,
# # #             "gender": input_data.gender,
# # #             "existing_conditions": input_data.existing_conditions,
# # #             "other_medications": input_data.other_medications,
# # #             "specialty": "drug"
# # #         }
        
# # #         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
# # #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Drug interaction error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/health/medical-term")
# # # async def explain_medical_term(input_data: MedicalTermInput):
# # #     """Explain medical terms with clarity."""
# # #     try:
# # #         if not input_data.term:
# # #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# # #         prompt = f"Explain the medical term: {input_data.term}"
# # #         if input_data.language and input_data.language != "en":
# # #             prompt += f" in {input_data.language} language"
        
# # #         context = {"specialty": "medical_term"}
# # #         result = await run_agent_with_thinking(medical_term_agent, prompt, context)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Medical term error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/health/report-summarize")
# # # async def summarize_medical_report(input_data: ReportTextInput):
# # #     """Summarize medical reports with intelligent analysis."""
# # #     try:
# # #         if not input_data.text:
# # #             raise HTTPException(status_code=400, detail="Report text is required")
        
# # #         prompt = f"""
# # #         Analyze and summarize this medical report:

# # #         {input_data.text}

# # #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# # #         Focus on key findings, recommendations, and next steps.
# # #         """
# # #         context = {"specialty": "report"}
# # #         result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Report summary error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/chatbot/session/clear")
# # # async def clear_session(request: ClearSessionRequest):
# # #     """Clear chatbot session history."""
# # #     try:
# # #         session_id = request.session_id or "default_session"
# # #         conn = get_db()
# # #         with conn.cursor() as cur:
# # #             cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
# # #             conn.commit()
# # #         conn.close()
# # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # #     except Exception as e:
# # #         logger.error(f"Clear session error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # @app.get("/health")
# # # async def health_check():
# # #     """Health check endpoint."""
# # #     return {
# # #         "status": "healthy",
# # #         "timestamp": datetime.now().isoformat(),
# # #         "version": "2.1.0",
# # #         "agents_available": True,
# # #         "thinking_enabled": True
# # #     }

# # # @app.get("/api/chatbot/sessions")
# # # async def get_sessions():
# # #     """Get active session count (for monitoring)."""
# # #     try:
# # #         conn = get_db()
# # #         with conn.cursor() as cur:
# # #             cur.execute("SELECT COUNT(*) FROM chat_sessions")
# # #             active_sessions = cur.fetchone()[0]
# # #             cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
# # #             total_messages_result = cur.fetchone()[0]
# # #             total_messages = total_messages_result if total_messages_result is not None else 0
# # #         conn.close()
# # #         return {
# # #             "active_sessions": active_sessions,
# # #             "total_messages": total_messages
# # #         }
# # #     except Exception as e:
# # #         logger.error(f"Get sessions error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Failed to retrieve session data")

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# import json
# import asyncio
# from datetime import datetime
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# from typing import Optional, Dict, List, Any
# import logging
# from contextlib import asynccontextmanager
# import uuid
# import httpx
# import re
# import pymysql

# # Import OpenAI Agents SDK components (assumed to be available)
# from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # Import agent creation functions from medicura_agents and specialist_agents
# from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
# from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
# from medicura_agents.general_health_agent import create_general_health_agent
# from medicura_agents.medical_term_agent import create_medical_term_agent
# from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
# from medicura_agents.about_agent import create_about_agent

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # TiDB Configuration with minimal SSL
# DB_CONFIG = {
#     "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
#     "port": 4000,
#     "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
#     "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
#     "database": os.getenv("TIDB_DATABASE", "test"),
#     "charset": "utf8mb4",
#     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
# }

# def get_db():
#     """Establish a connection to TiDB."""
#     try:
#         connection = pymysql.connect(**DB_CONFIG)
#         return connection
#     except pymysql.err.OperationalError as e:
#         logger.error(f"Failed to connect to TiDB: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manage application startup and shutdown."""
#     logger.info("Starting Medicura-AI Health Assistant")
#     logger.info("Connecting to TiDB...")
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS chat_sessions (
#                     session_id VARCHAR(100) PRIMARY KEY,
#                     history JSON NOT NULL,
#                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#                 )
#             """)
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS specialist_vectors (
#                     id VARCHAR(100) PRIMARY KEY,
#                     specialty VARCHAR(50) NOT NULL,
#                     content TEXT NOT NULL,
#                     embedding JSON NOT NULL,
#                     metadata JSON,
#                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                 )
#             """)
#             conn.commit()
#         conn.close()
#         logger.info("TiDB connected and tables ready.")
#         yield
#     except Exception as e:
#         logger.error(f"Lifespan error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Application startup failed")
#     finally:
#         logger.info("Shutting down Medicura-AI Health Assistant")

# app = FastAPI(
#     title="Medicura-AI Health Assistant",
#     description="AI-powered health assistant for symptom analysis and medical queries",
#     version="2.1.0",
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc"
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Environment Variable Validation
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY not found in environment variables")
#     raise ValueError("GEMINI_API_KEY environment variable is required")

# # AI Agent Initialization
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     http_client=httpx.AsyncClient(timeout=60.0)
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client,
# )

# model_settings = ModelSettings(
#     temperature=0.7,
#     top_p=0.9,
#     max_tokens=2048,
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     model_settings=model_settings,
#     tracing_disabled=True,
# )

# # Initialize Core Agents (Specialist agents ko temporarily comment out)
# symptom_analyzer_agent = create_symptom_analyzer_agent(model)
# drug_interaction_agent = create_drug_interaction_agent(model)
# general_health_agent = create_general_health_agent(model)
# medical_term_agent = create_medical_term_agent(model)
# report_analyzer_agent = create_report_analyzer_agent(model)
# about_agent = create_about_agent(model)

# # Temporary: Specialist agents ko comment out karo
# cardiology_agent = general_health_agent
# dermatology_agent = general_health_agent
# neurology_agent = general_health_agent
# pulmonology_agent = general_health_agent
# ophthalmology_agent = general_health_agent
# dental_agent = general_health_agent
# allergy_immunology_agent = general_health_agent
# pediatrics_agent = general_health_agent
# orthopedics_agent = general_health_agent
# mental_health_agent = general_health_agent
# endocrinology_agent = general_health_agent
# gastroenterology_agent = general_health_agent
# radiology_agent = general_health_agent
# infectious_disease_agent = general_health_agent
# vaccination_advisor_agent = general_health_agent

# def generate_embedding(text: str) -> List[float]:
#     """Generate embedding using Gemini API (simplified placeholder)."""
#     return [0.1] * 768  # Mock 768-dimensional embedding

# def ingest_specialty_data():
#     """Ingest sample data into specialist_vectors table."""
#     sample_data = [
#         {"id": "cardio_1", "specialty": "cardiology", "content": "chest pain, shortness of breath", "metadata": {"diagnosis": "Possible angina"}},
#         {"id": "cardio_2", "specialty": "cardiology", "content": "palpitations, fatigue", "metadata": {"diagnosis": "Possible arrhythmia"}},
#         {"id": "derm_1", "specialty": "dermatology", "content": "red itchy rash on arm", "metadata": {"diagnosis": "Possible eczema"}},
#         {"id": "derm_2", "specialty": "dermatology", "content": "dry skin patches", "metadata": {"diagnosis": "Possible psoriasis"}},
#         {"id": "neuro_1", "specialty": "neurology", "content": "headache, dizziness", "metadata": {"diagnosis": "Possible migraine"}},
#         {"id": "neuro_2", "specialty": "neurology", "content": "numbness in hands", "metadata": {"diagnosis": "Possible neuropathy"}},
#         {"id": "pulmo_1", "specialty": "pulmonology", "content": "persistent cough", "metadata": {"diagnosis": "Possible bronchitis"}},
#         {"id": "pulmo_2", "specialty": "pulmonology", "content": "wheezing, shortness of breath", "metadata": {"diagnosis": "Possible asthma"}},
#         {"id": "ophtha_1", "specialty": "ophthalmology", "content": "blurry vision", "metadata": {"diagnosis": "Possible cataract"}},
#         {"id": "dental_1", "specialty": "dental", "content": "toothache, swelling", "metadata": {"diagnosis": "Possible abscess"}},
#         {"id": "allergy_1", "specialty": "allergy_immunology", "content": "sneezing, runny nose", "metadata": {"diagnosis": "Possible allergies"}},
#         {"id": "peds_1", "specialty": "pediatrics", "content": "fever in child", "metadata": {"diagnosis": "Possible infection"}},
#         {"id": "ortho_1", "specialty": "orthopedics", "content": "joint pain", "metadata": {"diagnosis": "Possible arthritis"}},
#         {"id": "mental_1", "specialty": "mental_health", "content": "anxiety, stress", "metadata": {"diagnosis": "Possible anxiety disorder"}},
#         {"id": "endo_1", "specialty": "endocrinology", "content": "fatigue, weight gain", "metadata": {"diagnosis": "Possible hypothyroidism"}},
#         {"id": "gastro_1", "specialty": "gastroenterology", "content": "stomach pain", "metadata": {"diagnosis": "Possible gastritis"}},
#         {"id": "radio_1", "specialty": "radiology", "content": "abnormal x-ray", "metadata": {"diagnosis": "Possible fracture"}},
#         {"id": "infect_1", "specialty": "infectious_disease", "content": "fever, chills", "metadata": {"diagnosis": "Possible flu"}},
#         {"id": "vacc_1", "specialty": "vaccination_advisor", "content": "vaccine schedule", "metadata": {"recommendation": "Consult pediatrician"}}
#     ]
#     conn = get_db()
#     try:
#         with conn.cursor() as cur:
#             for item in sample_data:
#                 embedding = generate_embedding(item["content"])
#                 cur.execute("""
#                     INSERT INTO specialist_vectors (id, specialty, content, embedding, metadata)
#                     VALUES (%s, %s, %s, %s, %s)
#                     ON DUPLICATE KEY UPDATE content = VALUES(content), embedding = VALUES(embedding), metadata = VALUES(metadata)
#                 """, (item["id"], item["specialty"], item["content"], json.dumps(embedding), json.dumps(item["metadata"])))
#             conn.commit()
#     finally:
#         conn.close()

# # def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# #     """Search similar cases in specialist_vectors using cosine similarity."""
# #     embedding = generate_embedding(query)
# #     conn = get_db()
# #     try:
# #         with conn.cursor() as cur:
# #             cur.execute("""
# #                 SELECT id, content, metadata
# #                 FROM specialist_vectors
# #                 WHERE specialty = %s
# #                 ORDER BY (
# #                     1 - (
# #                         (SELECT SUM(a * b) FROM JSON_TABLE(
# #                             CAST(JSON_EXTRACT(embedding, '$') AS JSON),
# #                             '$[*]' COLUMNS (a DOUBLE PATH '$')
# #                         ) jt1,
# #                         JSON_TABLE(
# #                             %s,
# #                             '$[*]' COLUMNS (b DOUBLE PATH '$')
# #                         ) jt2) /
# #                         (SQRT((SELECT SUM(a * a) FROM JSON_TABLE(
# #                             CAST(JSON_EXTRACT(embedding, '$') AS JSON),
# #                             '$[*]' COLUMNS (a DOUBLE PATH '$')
# #                         ))) * SQRT((SELECT SUM(b * b) FROM JSON_TABLE(
# #                             %s,
# #                             '$[*]' COLUMNS (b DOUBLE PATH '$')
# #                         ))))
# #                     )
# #                 ) LIMIT %s
# #             """, (specialty, json.dumps(embedding), json.dumps(embedding), top_k))
# #             results = cur.fetchall()
# #             return [{"id": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]
# #     except Exception as e:
# #         logger.error(f"Vector search error: {str(e)}")
# #         return []
# #     finally:
# #         conn.close()

# def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
#     """Search similar cases in specialist_vectors using simple text matching."""
#     conn = get_db()
#     try:
#         with conn.cursor() as cur:
#             # Simple text-based search instead of complex vector search
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
#         logger.error(f"Search error: {str(e)}")
#         return []
#     finally:
#         conn.close()

# def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
#     """Extract JSON from agent response with multiple fallback methods."""
#     try:
#         try:
#             return json.loads(response.strip())
#         except json.JSONDecodeError:
#             pass
#         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group(1))
#         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
#         if brace_match:
#             return json.loads(brace_match.group(0))
#         return {
#             "summary": response,
#             "detailed_analysis": "Detailed analysis based on your query",
#             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
#             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
#             "type": "general"
#         }
#     except Exception as e:
#         logger.warning(f"JSON extraction failed: {str(e)}")
#         return None

# async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
#     """Run agent with enhanced thinking and robust error handling."""
#     try:
#         specialty = context.get("specialty", "general") if context else "general"

#         thinking_prompt = f"""
#         USER QUERY: {prompt}
#         CONTEXT: {json.dumps(context) if context else 'No additional context'}
#         SIMILAR CASES: {json.dumps(search_similar_cases(prompt, specialty))}
        
#         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
#         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
#         """
        
#         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
#         parsed_response = extract_json_from_response(result.final_output)
        
#         if parsed_response:
#             parsed_response.update({
#                 "timestamp": datetime.now().isoformat(),
#                 "success": True,
#                 "thinking_applied": True
#             })
#             return parsed_response
#         else:
#             return create_intelligent_response(result.final_output, prompt)
        
#     except Exception as e:
#         logger.error(f"Agent error: {str(e)}")
#         return create_intelligent_response(f"Analysis of: {prompt}")

# def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
#     """Create a well-structured response from text."""
#     return {
#         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
#         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
#         "recommendations": [
#             "Consult with a healthcare provider",
#             "Provide complete medical history for assessment",
#             "Follow evidence-based medical guidance"
#         ],
#         "when_to_seek_help": [
#             "Immediately for severe or emergency symptoms",
#             "Within 24-48 hours for persistent concerns",
#             "Routinely for preventive care"
#         ],
#         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
#         "type": "general",
#         "timestamp": datetime.now().isoformat(),
#         "success": True,
#         "thinking_applied": True
#     }

# def load_history(session_id: str) -> List[dict]:
#     """Load chat history from TiDB."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
#             result = cur.fetchone()
#         conn.close()
#         return json.loads(result[0]) if result else []
#     except Exception as e:
#         logger.error(f"Failed to load history: {str(e)}")
#         return []

# def save_history(session_id: str, history: List[dict]):
#     """Save chat history to TiDB."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("""
#                 INSERT INTO chat_sessions (session_id, history)
#                 VALUES (%s, %s)
#                 ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
#             """, (session_id, json.dumps(history), json.dumps(history)))
#             conn.commit()
#         conn.close()
#     except Exception as e:
#         logger.error(f"Failed to save history: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to save chat history")

# # Pydantic Models
# class ChatRequest(BaseModel):
#     message: str = Field(..., min_length=1, max_length=1000)
#     session_id: Optional[str] = Field(None, max_length=100)
#     context: Optional[dict] = None

# class DrugInteractionInput(BaseModel):
#     medications: List[str] = Field(..., min_items=1, max_items=10)
#     age: Optional[int] = Field(None, ge=0, le=120)
#     gender: Optional[str] = Field(None, max_length=20)
#     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
#     other_medications: Optional[List[str]] = Field(None, max_items=20)

# class MedicalTermInput(BaseModel):
#     term: str = Field(..., min_length=1, max_length=100)
#     language: Optional[str] = Field("en", max_length=10)

# class ReportTextInput(BaseModel):
#     text: str = Field(..., min_length=10, max_length=10000)
#     language: Optional[str] = Field("en", max_length=10)

# class ClearSessionRequest(BaseModel):
#     session_id: Optional[str] = Field(None, max_length=100)

# # API Endpoints
# @app.post("/api/chatbot")
# async def chatbot(request: ChatRequest):
#     """Main chatbot endpoint with intelligent thinking and specialty support."""
#     try:
#         session_id = request.session_id or str(uuid.uuid4())
#         history = load_history(session_id)

#         # Ingest specialty data on first run
#         if not history:
#             ingest_specialty_data()

#         # Select appropriate agent based on specialty and keywords
#         query_lower = request.message.lower()
#         specialty_map = {
#             "symptom": ["symptom", "pain", "fever", "headache", "nausea", "ache", "hurt"],
#             "drug": ["drug", "medication", "pill", "dose", "interaction", "side effect", "ibuprofen", "glutathion"],
#             "medical_term": ["what is", "explain", "define", "meaning of"],
#             "report": ["report", "result", "test", "lab", "x-ray", "summary"],
#             "about": ["creator", "author", "hadiqa", "gohar", "medicura about", "who made"]
#         }

#         selected_specialty = "general"
#         selected_agent = general_health_agent
        
#         for specialty, keywords in specialty_map.items():
#             if any(keyword in query_lower for keyword in keywords):
#                 selected_specialty = specialty
#                 if specialty == "symptom":
#                     selected_agent = symptom_analyzer_agent
#                 elif specialty == "drug":
#                     selected_agent = drug_interaction_agent
#                 elif specialty == "medical_term":
#                     selected_agent = medical_term_agent
#                 elif specialty == "report":
#                     selected_agent = report_analyzer_agent
#                 elif specialty == "about":
#                     selected_agent = about_agent
#                 break

#         # Run agent with thinking mode and vector search context
#         context = request.context or {}
#         context["specialty"] = selected_specialty
#         context["similar_cases"] = search_similar_cases(request.message, selected_specialty)
#         result = await run_agent_with_thinking(selected_agent, request.message, context)

#         # Update chat history
#         history.extend([
#             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
#             {"role": "assistant", "content": json.dumps(result), "timestamp": datetime.now().isoformat()}
#         ])
#         history = history[-20:]  # Keep last 20 messages
#         save_history(session_id, history)

#         return JSONResponse(content=result)

#     except Exception as e:
#         logger.error(f"Chatbot error: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
#         )

# @app.post("/api/health/drug-interactions")
# async def check_drug_interactions(input_data: DrugInteractionInput):
#     """Check drug interactions with thorough analysis."""
#     try:
#         if not input_data.medications or len(input_data.medications) == 0:
#             raise HTTPException(status_code=400, detail="At least one medication is required")
        
#         context = {
#             "medications": input_data.medications,
#             "age": input_data.age,
#             "gender": input_data.gender,
#             "existing_conditions": input_data.existing_conditions,
#             "other_medications": input_data.other_medications,
#             "specialty": "drug"
#         }
        
#         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
#         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Drug interaction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/medical-term")
# async def explain_medical_term(input_data: MedicalTermInput):
#     """Explain medical terms with clarity."""
#     try:
#         if not input_data.term:
#             raise HTTPException(status_code=400, detail="Medical term is required")
        
#         prompt = f"Explain the medical term: {input_data.term}"
#         if input_data.language and input_data.language != "en":
#             prompt += f" in {input_data.language} language"
        
#         context = {"specialty": "medical_term"}
#         result = await run_agent_with_thinking(medical_term_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Medical term error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/report-summarize")
# async def summarize_medical_report(input_data: ReportTextInput):
#     """Summarize medical reports with intelligent analysis."""
#     try:
#         if not input_data.text:
#             raise HTTPException(status_code=400, detail="Report text is required")
        
#         prompt = f"""
#         Analyze and summarize this medical report:

#         {input_data.text}

#         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
#         Focus on key findings, recommendations, and next steps.
#         """
#         context = {"specialty": "report"}
#         result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Report summary error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/chatbot/session/clear")
# async def clear_session(request: ClearSessionRequest):
#     """Clear chatbot session history."""
#     try:
#         session_id = request.session_id or "default_session"
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
#             conn.commit()
#         conn.close()
#         return {"message": "Session cleared successfully", "session_id": session_id}
#     except Exception as e:
#         logger.error(f"Clear session error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to clear session")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": "2.1.0",
#         "agents_available": True,
#         "thinking_enabled": True
#     }

# @app.get("/api/chatbot/sessions")
# async def get_sessions():
#     """Get active session count (for monitoring)."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("SELECT COUNT(*) FROM chat_sessions")
#             active_sessions = cur.fetchone()[0]
#             cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
#             total_messages_result = cur.fetchone()[0]
#             total_messages = total_messages_result if total_messages_result is not None else 0
#         conn.close()
#         return {
#             "active_sessions": active_sessions,
#             "total_messages": total_messages
#         }
#     except Exception as e:
#         logger.error(f"Get sessions error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve session data")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)







# # ----------------- Start wroking for vector search 


# import os
# import json
# import asyncio
# from datetime import datetime
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# from typing import Optional, Dict, List, Any
# import logging
# from contextlib import asynccontextmanager
# import uuid
# import httpx
# import re
# import pymysql
# import google.generativeai as genai 


# # In main.py and cardiology_ai.py
# from utils import search_similar_cases, fallback_text_search
# # Import OpenAI Agents SDK components (assumed to be available)
# from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # Import agent creation functions from medicura_agents and specialist_agents
# from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
# from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
# from medicura_agents.general_health_agent import create_general_health_agent
# from medicura_agents.medical_term_agent import create_medical_term_agent
# from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
# from medicura_agents.about_agent import create_about_agent


# from specialist_agents.cardiology_ai import create_cardiology_agent
# from specialist_agents.dermatology_ai import create_dermatology_agent
# from specialist_agents.neurology_ai import create_neurology_agent
# from specialist_agents.pulmonology_ai import create_pulmonology_agent
# from specialist_agents.ophthalmology_ai import create_ophthalmology_agent
# from specialist_agents.dental_ai import create_dental_agent
# from specialist_agents.allergy_immunology_ai import create_allergy_immunology_agent
# from specialist_agents.pediatrics_ai import create_pediatrics_agent
# from specialist_agents.orthopedics_ai import create_orthopedics_agent
# from specialist_agents.mental_health_ai import create_mental_health_agent
# from specialist_agents.endocrinology_ai import create_endocrinology_agent
# from specialist_agents.gastroenterology_ai import create_gastroenterology_agent
# from specialist_agents.radiology_ai import create_radiology_agent
# from specialist_agents.infectious_disease_ai import create_infectious_disease_agent
# from specialist_agents.vaccination_advisor_ai import create_vaccination_advisor_agent
# from specialist_agents.drug_interaction_agent import create_drug_interaction_agent


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()


# # TiDB Configuration with minimal SSL
# DB_CONFIG = {
#     "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
#     "port": 4000,
#     "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
#     "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
#     "database": os.getenv("TIDB_DATABASE", "test"),
#     "charset": "utf8mb4",
#     "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
# }

# def get_db():
#     """Establish a connection to TiDB."""
#     try:
#         connection = pymysql.connect(**DB_CONFIG)
#         return connection
#     except pymysql.err.OperationalError as e:
#         logger.error(f"Failed to connect to TiDB: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     """Manage application startup and shutdown."""
# #     logger.info("Starting Medicura-AI Health Assistant")
# #     logger.info("Connecting to TiDB...")
# #     try:
# #         conn = get_db()
# #         with conn.cursor() as cur:
# #             cur.execute("""
# #                 CREATE TABLE IF NOT EXISTS chat_sessions (
# #                     session_id VARCHAR(100) PRIMARY KEY,
# #                     history JSON NOT NULL,
# #                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# #                 )
# #             """)
# #             cur.execute("""
# #                 CREATE TABLE IF NOT EXISTS specialist_vectors (
# #                     id VARCHAR(100) PRIMARY KEY,
# #                     specialty VARCHAR(50) NOT NULL,
# #                     content TEXT NOT NULL,
# #                     embedding JSON NOT NULL,
# #                     metadata JSON,
# #                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# #                 )
# #             """)
# #             conn.commit()
# #         conn.close()
# #         logger.info("TiDB connected and tables ready.")
# #         yield
# #     except Exception as e:
# #         logger.error(f"Lifespan error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Application startup failed")
# #     finally:
# #         logger.info("Shutting down Medicura-AI Health Assistant")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manage application startup and shutdown."""
#     logger.info("Starting Medicura-AI Health Assistant")
#     logger.info("Connecting to TiDB...")
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS chat_sessions (
#                     session_id VARCHAR(100) PRIMARY KEY,
#                     history JSON NOT NULL,
#                     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#                 )
#             """)
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS specialist_vectors (
#                     id VARCHAR(100) PRIMARY KEY,
#                     specialty VARCHAR(50) NOT NULL,
#                     content TEXT NOT NULL,
#                     embedding VECTOR(768) NOT NULL,  -- CHANGED FROM JSON TO VECTOR(768)
#                     metadata JSON,
#                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                 )
#             """)
#             conn.commit()
#         conn.close()
#         logger.info("TiDB connected and tables ready.")
#         yield
#     except Exception as e:
#         logger.error(f"Lifespan error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Application startup failed")
#     finally:
#         logger.info("Shutting down Medicura-AI Health Assistant")

# app = FastAPI(
#     title="Medicura-AI Health Assistant",
#     description="AI-powered health assistant for symptom analysis and medical queries",
#     version="2.1.0",
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc"
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Environment Variable Validation
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY not found in environment variables")
#     raise ValueError("GEMINI_API_KEY environment variable is required")


# # AI Agent Initialization
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     http_client=httpx.AsyncClient(timeout=60.0)
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client,
# )

# model_settings = ModelSettings(
#     temperature=0.7,
#     top_p=0.9,
#     max_tokens=2048,
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     model_settings=model_settings,
#     tracing_disabled=True,
# )

# # Initialize Core Agents (Specialist agents ko temporarily comment out)
# symptom_analyzer_agent = create_symptom_analyzer_agent(model)
# drug_interaction_agent = create_drug_interaction_agent(model)
# general_health_agent = create_general_health_agent(model)
# medical_term_agent = create_medical_term_agent(model)
# report_analyzer_agent = create_report_analyzer_agent(model)
# about_agent = create_about_agent(model)


# # Specialist agents
# cardiology_agent = create_cardiology_agent(model)
# dermatology_agent = create_dermatology_agent(model)
# neurology_agent = create_neurology_agent(model)
# pulmonology_agent = create_pulmonology_agent(model)
# ophthalmology_agent = create_ophthalmology_agent(model)
# dental_agent = create_dental_agent(model)
# allergy_immunology_agent = create_allergy_immunology_agent(model)
# pediatrics_agent = create_pediatrics_agent(model)
# orthopedics_agent = create_orthopedics_agent(model)
# mental_health_agent = create_mental_health_agent(model)
# endocrinology_agent = create_endocrinology_agent(model)
# gastroenterology_agent = create_gastroenterology_agent(model)
# radiology_agent = create_radiology_agent(model)
# infectious_disease_agent = create_infectious_disease_agent(model)
# vaccination_advisor_agent = create_vaccination_advisor_agent(model)
# drug_interaction_agent = create_drug_interaction_agent(model)


# # def generate_embedding(text: str) -> List[float]:
# #     """Generate embedding using Gemini API (simplified placeholder)."""
# #     return [0.1] * 768  # Mock 768-dimensional embedding

# # Configure the Gemini client (add this near your other config code, e.g., after loading env vars)
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# def generate_embedding(text: str) -> List[float]:
#     """Generate a real embedding vector using the Gemini embedding model."""
#     try:
#         # Call the Gemini Embedding API
#         result = genai.embed_content(
#             model="models/embedding-001",
#             content=text,
#             task_type="retrieval_document" # Or "retrieval_query", "classification", etc.
#         )
#         return result['embedding']
#     except Exception as e:
#         logger.error(f"Embedding generation failed: {str(e)}")
#         # Fallback to avoid breaking the application, but log the error heavily.
#         return [0.0] * 768

# # def ingest_specialty_data():
# #     """Ingest sample data into specialist_vectors table."""
# #     sample_data = [
# #         {"id": "cardio_1", "specialty": "cardiology", "content": "chest pain, shortness of breath", "metadata": {"diagnosis": "Possible angina"}},
# #         {"id": "cardio_2", "specialty": "cardiology", "content": "palpitations, fatigue", "metadata": {"diagnosis": "Possible arrhythmia"}},
# #         {"id": "derm_1", "specialty": "dermatology", "content": "red itchy rash on arm", "metadata": {"diagnosis": "Possible eczema"}},
# #         {"id": "derm_2", "specialty": "dermatology", "content": "dry skin patches", "metadata": {"diagnosis": "Possible psoriasis"}},
# #         {"id": "neuro_1", "specialty": "neurology", "content": "headache, dizziness", "metadata": {"diagnosis": "Possible migraine"}},
# #         {"id": "neuro_2", "specialty": "neurology", "content": "numbness in hands", "metadata": {"diagnosis": "Possible neuropathy"}},
# #         {"id": "pulmo_1", "specialty": "pulmonology", "content": "persistent cough", "metadata": {"diagnosis": "Possible bronchitis"}},
# #         {"id": "pulmo_2", "specialty": "pulmonology", "content": "wheezing, shortness of breath", "metadata": {"diagnosis": "Possible asthma"}},
# #         {"id": "ophtha_1", "specialty": "ophthalmology", "content": "blurry vision", "metadata": {"diagnosis": "Possible cataract"}},
# #         {"id": "dental_1", "specialty": "dental", "content": "toothache, swelling", "metadata": {"diagnosis": "Possible abscess"}},
# #         {"id": "allergy_1", "specialty": "allergy_immunology", "content": "sneezing, runny nose", "metadata": {"diagnosis": "Possible allergies"}},
# #         {"id": "peds_1", "specialty": "pediatrics", "content": "fever in child", "metadata": {"diagnosis": "Possible infection"}},
# #         {"id": "ortho_1", "specialty": "orthopedics", "content": "joint pain", "metadata": {"diagnosis": "Possible arthritis"}},
# #         {"id": "mental_1", "specialty": "mental_health", "content": "anxiety, stress", "metadata": {"diagnosis": "Possible anxiety disorder"}},
# #         {"id": "endo_1", "specialty": "endocrinology", "content": "fatigue, weight gain", "metadata": {"diagnosis": "Possible hypothyroidism"}},
# #         {"id": "gastro_1", "specialty": "gastroenterology", "content": "stomach pain", "metadata": {"diagnosis": "Possible gastritis"}},
# #         {"id": "radio_1", "specialty": "radiology", "content": "abnormal x-ray", "metadata": {"diagnosis": "Possible fracture"}},
# #         {"id": "infect_1", "specialty": "infectious_disease", "content": "fever, chills", "metadata": {"diagnosis": "Possible flu"}},
# #         {"id": "vacc_1", "specialty": "vaccination_advisor", "content": "vaccine schedule", "metadata": {"recommendation": "Consult pediatrician"}}
# #     ]
# #     conn = get_db()
# #     try:
# #         with conn.cursor() as cur:
# #             for item in sample_data:
# #                 embedding = generate_embedding(item["content"])
# #                 cur.execute("""
# #                     INSERT INTO specialist_vectors (id, specialty, content, embedding, metadata)
# #                     VALUES (%s, %s, %s, %s, %s)
# #                     ON DUPLICATE KEY UPDATE content = VALUES(content), embedding = VALUES(embedding), metadata = VALUES(metadata)
# #                 """, (item["id"], item["specialty"], item["content"], json.dumps(embedding), json.dumps(item["metadata"])))
# #             conn.commit()
# #     finally:
# #         conn.close()

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

# # # def search_similar_cases(query: str, specialty: str, top_k: int = 5) -> List[Dict]:
# # #     """Search similar cases in specialist_vectors using simple text matching."""
# # #     conn = get_db()
# # #     try:
# # #           # Generate embedding for the query
# # #         query_embedding = generate_embedding(query)
        
# # #         with conn.cursor() as cur:
# # #             # Simple text-based search instead of complex vector search
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
# # #         logger.error(f"Search error: {str(e)}")
# # #         return []
# # #     finally:
# # #         conn.close()

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

# def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
#     """Extract JSON from agent response with multiple fallback methods."""
#     try:
#         try:
#             return json.loads(response.strip())
#         except json.JSONDecodeError:
#             pass
#         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group(1))
#         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
#         if brace_match:
#             return json.loads(brace_match.group(0))
#         return {
#             "summary": response,
#             "detailed_analysis": "Detailed analysis based on your query",
#             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
#             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
#             "type": "general"
#         }
#     except Exception as e:
#         logger.warning(f"JSON extraction failed: {str(e)}")
#         return None

# # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# #     """Run agent with enhanced thinking and robust error handling."""
# #     try:
# #         specialty = context.get("specialty", "general") if context else "general"

# #         thinking_prompt = f"""
# #         USER QUERY: {prompt}
# #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
# #         SIMILAR CASES: {json.dumps(search_similar_cases(prompt, specialty))}
        
# #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# #         """
        
# #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# #         parsed_response = extract_json_from_response(result.final_output)
        
# #         if parsed_response:
# #             parsed_response.update({
# #                 "timestamp": datetime.now().isoformat(),
# #                 "success": True,
# #                 "thinking_applied": True
# #             })
# #             return parsed_response
# #         else:
# #             return create_intelligent_response(result.final_output, prompt)
        
# #     except Exception as e:
# #         logger.error(f"Agent error: {str(e)}")
# #         return create_intelligent_response(f"Analysis of: {prompt}")


# async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
#     """Run agent with enhanced thinking and robust error handling."""
#     try:
#         specialty = context.get("specialty", "general") if context else "general"
        
#         # For drug-related queries, provide more specific context
#         if specialty == "drug":
#             thinking_prompt = f"""
#             USER QUERY: {prompt}
#             CONTEXT: This is a drug-related query. Please provide information about usage, dosage, precautions, and interactions.
            
#             PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
#             DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
#             """

#         elif specialty == "symptom":
#             thinking_prompt = f"""
#             USER QUERY: {prompt}
#             CONTEXT: This is a symptom analysis query. Provide comprehensive information about 
#             possible causes, self-care measures, when to seek help, and warning signs.
            
#             RESPONSE FORMAT: Provide a comprehensive JSON response with detailed fields.
#             """

#         else:
#             thinking_prompt = f"""
#             USER QUERY: {prompt}
#             CONTEXT: {json.dumps(context) if context else 'No additional context'}
            
#             PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
#             DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
#             """
        
#         result = await Runner.run(agent, thinking_prompt, run_config=config)
#         logger.info(f"Raw agent response: {result.final_output[:200]}...")

        
#         parsed_response = extract_json_from_response(result.final_output)
        
#         if parsed_response:
#             parsed_response.update({
#                 "timestamp": datetime.now().isoformat(),
#                 "success": True,
#                 "thinking_applied": True
#             })
#             return parsed_response
#         else:
#             return create_intelligent_response(result.final_output, prompt, specialty)
        
#     except Exception as e:
#         logger.error(f"Agent error: {str(e)}")
#         # Create a fallback response based on the query
#         if "headache" in prompt.lower() and "panadol" in prompt.lower():
#             return {
#                 "summary": "Panadol (paracetamol) can generally be taken for headaches",
#                 "detailed_analysis": "Panadol (paracetamol) is commonly used for headache relief. The typical adult dosage is 500-1000mg every 4-6 hours as needed, not exceeding 4000mg in 24 hours. Make sure you don't have any contraindications like liver disease.",
#                 "recommendations": [
#                     "Follow dosage instructions on packaging",
#                     "Don't exceed maximum daily dose",
#                     "Consult doctor if headache persists beyond 3 days"
#                 ],
#                 "when_to_seek_help": [
#                     "If headache is severe or sudden",
#                     "If accompanied by fever, stiff neck, or vision changes",
#                     "If headache persists despite medication"
#                 ],
#                 "disclaimer": "This is general information. Consult healthcare professionals for personalized advice.",
#                 "type": "drug",
#                 "timestamp": datetime.now().isoformat(),
#                 "success": True
#             }
#         return create_intelligent_response(f"Analysis of: {prompt}")
    

# def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
#     """Create a well-structured response from text."""
#     return {
#         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
#         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
#         "recommendations": [
#             "Consult with a healthcare provider",
#             "Provide complete medical history for assessment",
#             "Follow evidence-based medical guidance"
#         ],
#         "when_to_seek_help": [
#             "Immediately for severe or emergency symptoms",
#             "Within 24-48 hours for persistent concerns",
#             "Routinely for preventive care"
#         ],
#         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
#         "type": "general",
#         "timestamp": datetime.now().isoformat(),
#         "success": True,
#         "thinking_applied": True
#     }

# # -----
# def create_structured_response_from_text(text: str, original_query: str, specialty: str) -> Dict[str, Any]:
#     """Create a structured response when JSON parsing fails."""
#     base_response = {
#         "summary": text[:150] + "..." if len(text) > 150 else text,
#         "detailed_analysis": text,
#         "timestamp": datetime.now().isoformat(),
#         "success": True,
#         "thinking_applied": True
#     }
    
#     # Add specialty-specific fields
#     if specialty == "drug":
#         base_response.update({
#             "type": "drug",
#             "recommendations": ["Follow dosage instructions", "Consult doctor if unsure", "Read medication leaflet"],
#             "disclaimer": "This is general information. Consult healthcare professionals for personalized advice."
#         })
#     elif specialty == "symptom":
#         base_response.update({
#             "type": "symptom",
#             "when_to_seek_help": ["If symptoms persist", "If severe pain", "If symptoms worsen"],
#             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice."
#         })
    
#     return base_response

# async def run_multi_agent_workflow(prompt: str, context: Dict = None):
#     """Chain multiple agents for comprehensive analysis."""
#     # Symptom → Drug → General Health chain
#     symptom_result = await Runner.run(symptom_analyzer_agent, prompt, run_config=config)
#     drug_result = await Runner.run(drug_interaction_agent, f"Symptoms: {prompt}\nAnalysis: {symptom_result.final_output}", run_config=config)
#     health_result = await Runner.run(general_health_agent, f"Symptoms: {prompt}\nDrug Analysis: {drug_result.final_output}", run_config=config)
    
#     return {
#         "symptom_analysis": extract_json_from_response(symptom_result.final_output),
#         "drug_analysis": extract_json_from_response(drug_result.final_output),
#         "health_analysis": extract_json_from_response(health_result.final_output),
#         "multi_agent_workflow": True
#     }

# def load_history(session_id: str) -> List[dict]:
#     """Load chat history from TiDB."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
#             result = cur.fetchone()
#         conn.close()
#         return json.loads(result[0]) if result else []
#     except Exception as e:
#         logger.error(f"Failed to load history: {str(e)}")
#         return []

# def save_history(session_id: str, history: List[dict]):
#     """Save chat history to TiDB."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("""
#                 INSERT INTO chat_sessions (session_id, history)
#                 VALUES (%s, %s)
#                 ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
#             """, (session_id, json.dumps(history), json.dumps(history)))
#             conn.commit()
#         conn.close()
#     except Exception as e:
#         logger.error(f"Failed to save history: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to save chat history")

# # Pydantic Models
# class ChatRequest(BaseModel):
#     message: str = Field(..., min_length=1, max_length=1000)
#     session_id: Optional[str] = Field(None, max_length=100)
#     context: Optional[dict] = None

# class DrugInteractionInput(BaseModel):
#     medications: List[str] = Field(..., min_items=1, max_items=10)
#     age: Optional[int] = Field(None, ge=0, le=120)
#     gender: Optional[str] = Field(None, max_length=20)
#     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
#     other_medications: Optional[List[str]] = Field(None, max_items=20)

# class MedicalTermInput(BaseModel):
#     term: str = Field(..., min_length=1, max_length=100)
#     language: Optional[str] = Field("en", max_length=10)

# class ReportTextInput(BaseModel):
#     text: str = Field(..., min_length=10, max_length=10000)
#     language: Optional[str] = Field("en", max_length=10)

# class ClearSessionRequest(BaseModel):
#     session_id: Optional[str] = Field(None, max_length=100)

# # # API Endpoints
# # @app.post("/api/chatbot")
# # async def chatbot(request: ChatRequest):
# #     """Main chatbot endpoint with intelligent thinking and specialty support."""
# #     try:
# #         session_id = request.session_id or str(uuid.uuid4())
# #         history = load_history(session_id)

# #         # Ingest specialty data on first run
# #         # if not history:
# #         #     ingest_specialty_data()

# #         # Select appropriate agent based on specialty and keywords
# #         query_lower = request.message.lower()
       
# #         specialty_map = {
# #             "symptom": ["symptom", "pain", "fever", "headache", "nausea", "ache", "hurt"],
# #             "drug": ["drug", "medication", "pill", "dose", "interaction", "side effect", "ibuprofen", "glutathion"],
# #             "medical_term": ["what is", "explain", "define", "meaning of"],
# #             "report": ["report", "result", "test", "lab", "x-ray", "summary"],
# #             "about": ["creator", "author", "hadiqa", "gohar", "medicura about", "who made"],
# #             "cardiology": ["heart", "cardio", "chest pain", "palpitations"],
# #             "dermatology": ["skin", "rash", "eczema", "psoriasis"],
# #             "neurology": ["brain", "migraine", "seizure", "numbness"],
# #             "pulmonology": ["lung", "cough", "asthma", "bronchitis"],
# #             "ophthalmology": ["eye", "vision", "blurred vision", "cataract"],
# #             "dental": ["tooth", "dentist", "toothache", "gum"],
# #             "allergy_immunology": ["allergy", "sneeze", "immunology", "pollen"],
# #             "pediatrics": ["child", "baby", "infant", "pediatric"],
# #             "orthopedics": ["bone", "joint", "fracture", "arthritis"],
# #             "mental_health": ["mental", "stress", "depression", "anxiety"],
# #             "endocrinology": ["hormone", "thyroid", "diabetes", "endocrine"],
# #             "gastroenterology": ["stomach", "abdomen", "gastritis", "ulcer"],
# #             "radiology": ["x-ray", "mri", "ct scan", "radiology"],
# #             "infectious_disease": ["flu", "infection", "virus", "bacteria"],
# #             "vaccination_advisor": ["vaccine", "immunization", "vaccination"]
# # }


# #         selected_specialty = "general"
# #         selected_agent = general_health_agent

# #         for specialty, keywords in specialty_map.items():
# #             if any(keyword in query_lower for keyword in keywords):
# #                 selected_specialty = specialty
# #                 if specialty == "symptom":
# #                     selected_agent = symptom_analyzer_agent
# #                 elif specialty == "drug":
# #                     selected_agent = drug_interaction_agent
# #                 elif specialty == "medical_term":
# #                     selected_agent = medical_term_agent
# #                 elif specialty == "report":
# #                     selected_agent = report_analyzer_agent
# #                 elif specialty == "about":
# #                     selected_agent = about_agent
# #                 elif specialty == "cardiology":
# #                     selected_agent = cardiology_agent
# #                 elif specialty == "dermatology":
# #                     selected_agent = dermatology_agent
# #                 elif specialty == "neurology":
# #                     selected_agent = neurology_agent
# #                 elif specialty == "pulmonology":
# #                     selected_agent = pulmonology_agent
# #                 elif specialty == "ophthalmology":
# #                     selected_agent = ophthalmology_agent
# #                 elif specialty == "dental":
# #                     selected_agent = dental_agent
# #                 elif specialty == "allergy_immunology":
# #                     selected_agent = allergy_immunology_agent
# #                 elif specialty == "pediatrics":
# #                     selected_agent = pediatrics_agent
# #                 elif specialty == "orthopedics":
# #                     selected_agent = orthopedics_agent
# #                 elif specialty == "mental_health":
# #                     selected_agent = mental_health_agent
# #                 elif specialty == "endocrinology":
# #                     selected_agent = endocrinology_agent
# #                 elif specialty == "gastroenterology":
# #                     selected_agent = gastroenterology_agent
# #                 elif specialty == "radiology":
# #                     selected_agent = radiology_agent
# #                 elif specialty == "infectious_disease":
# #                     selected_agent = infectious_disease_agent
# #                 elif specialty == "vaccination_advisor":
# #                     selected_agent = vaccination_advisor_agent
# #                 break  # ab sirf ek agent select hoga


# #         # Run agent with thinking mode and vector search context
# #         # context = request.context or {}
# #         # context["specialty"] = selected_specialty
# #         # context["similar_cases"] = search_similar_cases(request.message, selected_specialty)
# #         # result = await run_agent_with_thinking(selected_agent, request.message, context)
# #         # Run agent with thinking mode and vector search context
# #         context = request.context or {}
# #         context["specialty"] = selected_specialty
# #         context["similar_cases"] = search_similar_cases(request.message, selected_specialty)

# #         # Use multi-agent workflow for symptom analysis, single agent for others
# #         if selected_specialty == "symptom":
# #             result = await run_multi_agent_workflow(request.message, context)
# #         else:
# #             result = await run_agent_with_thinking(selected_agent, request.message, context)

# #             # Update chat history
# #         history.extend([
# #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# #             {"role": "assistant", "content": json.dumps(result), "timestamp": datetime.now().isoformat()}
# #         ])
# #         history = history[-20:]  # Keep last 20 messages
# #         save_history(session_id, history)

# #         return JSONResponse(content=result)

# #     except Exception as e:
# #         logger.error(f"Chatbot error: {str(e)}")
# #         return JSONResponse(
# #             status_code=500,
# #             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
# #         )
    
# @app.post("/api/chatbot")
# async def chatbot(request: ChatRequest):
#     """Main chatbot endpoint with intelligent thinking and specialty support."""
#     try:
#         session_id = request.session_id or str(uuid.uuid4())
#         history = load_history(session_id)

#         # Select appropriate agent based on specialty and keywords
#         query_lower = request.message.lower()
#         logger.info(f"Received query: {query_lower}")
       
#         specialty_map = {
#             "symptom": ["symptom", "pain", "fever", "headache", "nausea", "ache", "hurt"],
#             "drug": ["drug", "medication", "pill", "dose", "interaction", "side effect", "ibuprofen", "panadol", "paracetamol"],
#             "medical_term": ["what is", "explain", "define", "meaning of"],
#             "report": ["report", "result", "test", "lab", "x-ray", "summary"],
#             "about": ["creator", "author", "hadiqa", "gohar", "medicura about", "who made"],
#             "cardiology": ["heart", "cardio", "chest pain", "palpitations"],
#             "dermatology": ["skin", "rash", "eczema", "psoriasis"],
#             "neurology": ["brain", "migraine", "seizure", "numbness"],
#             "pulmonology": ["lung", "cough", "asthma", "bronchitis"],
#             "ophthalmology": ["eye", "vision", "blurred vision", "cataract"],
#             "dental": ["tooth", "dentist", "toothache", "gum"],
#             "allergy_immunology": ["allergy", "sneeze", "immunology", "pollen"],
#             "pediatrics": ["child", "baby", "infant", "pediatric"],
#             "orthopedics": ["bone", "joint", "fracture", "arthritis"],
#             "mental_health": ["mental", "stress", "depression", "anxiety"],
#             "endocrinology": ["hormone", "thyroid", "diabetes", "endocrine"],
#             "gastroenterology": ["stomach", "abdomen", "gastritis", "ulcer"],
#             "radiology": ["x-ray", "mri", "ct scan", "radiology"],
#             "infectious_disease": ["flu", "infection", "virus", "bacteria"],
#             "vaccination_advisor": ["vaccine", "immunization", "vaccination"]
#         }

#         selected_specialty = "general"
#         selected_agent = general_health_agent

#         for specialty, keywords in specialty_map.items():
#             if any(keyword in query_lower for keyword in keywords):
#                 selected_specialty = specialty
#                 logger.info(f"Selected specialty: {specialty}")
                
#                 # Map specialties to agents
#                 agent_mapping = {
#                     "symptom": symptom_analyzer_agent,
#                     "drug": drug_interaction_agent,
#                     "medical_term": medical_term_agent,
#                     "report": report_analyzer_agent,
#                     "about": about_agent,
#                     "cardiology": cardiology_agent,
#                     "dermatology": dermatology_agent,
#                     "neurology": neurology_agent,
#                     "pulmonology": pulmonology_agent,
#                     "ophthalmology": ophthalmology_agent,
#                     "dental": dental_agent,
#                     "allergy_immunology": allergy_immunology_agent,
#                     "pediatrics": pediatrics_agent,
#                     "orthopedics": orthopedics_agent,
#                     "mental_health": mental_health_agent,
#                     "endocrinology": endocrinology_agent,
#                     "gastroenterology": gastroenterology_agent,
#                     "radiology": radiology_agent,
#                     "infectious_disease": infectious_disease_agent,
#                     "vaccination_advisor": vaccination_advisor_agent
#                 }
                
#                 selected_agent = agent_mapping.get(specialty, general_health_agent)
#                 break

#         logger.info(f"Final selected agent: {selected_specialty}")

#         # Run agent with thinking mode
#         context = {"specialty": selected_specialty}
#         result = await run_agent_with_thinking(selected_agent, request.message, context)

#         # Update chat history
#         history.extend([
#             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
#             {"role": "assistant", "content": json.dumps(result), "timestamp": datetime.now().isoformat()}
#         ])
#         history = history[-20:]  # Keep last 20 messages
#         save_history(session_id, history)

#         return JSONResponse(content=result)

#     except Exception as e:
#         logger.error(f"Chatbot error: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
#         )

# @app.get("/api/test/vector-search")
# async def test_vector_search(query: str = "chest pain", specialty: str = "cardiology"):
#     """Test endpoint for vector search functionality."""
#     try:
#         results = search_similar_cases(query, specialty)
#         return {
#             "query": query,
#             "specialty": specialty, 
#             "results": results,
#             "vector_search_working": True
#         }
#     except Exception as e:
#         return {"error": str(e), "vector_search_working": False}    

# @app.post("/api/health/drug-interactions")
# async def check_drug_interactions(input_data: DrugInteractionInput):
#     """Check drug interactions with thorough analysis."""
#     try:
#         if not input_data.medications or len(input_data.medications) == 0:
#             raise HTTPException(status_code=400, detail="At least one medication is required")
        
#         context = {
#             "medications": input_data.medications,
#             "age": input_data.age,
#             "gender": input_data.gender,
#             "existing_conditions": input_data.existing_conditions,
#             "other_medications": input_data.other_medications,
#             "specialty": "drug"
#         }
        
#         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
#         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Drug interaction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/medical-term")
# async def explain_medical_term(input_data: MedicalTermInput):
#     """Explain medical terms with clarity."""
#     try:
#         if not input_data.term:
#             raise HTTPException(status_code=400, detail="Medical term is required")
        
#         prompt = f"Explain the medical term: {input_data.term}"
#         if input_data.language and input_data.language != "en":
#             prompt += f" in {input_data.language} language"
        
#         context = {"specialty": "medical_term"}
#         result = await run_agent_with_thinking(medical_term_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Medical term error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/report-summarize")
# async def summarize_medical_report(input_data: ReportTextInput):
#     """Summarize medical reports with intelligent analysis."""
#     try:
#         if not input_data.text:
#             raise HTTPException(status_code=400, detail="Report text is required")
        
#         prompt = f"""
#         Analyze and summarize this medical report:

#         {input_data.text}

#         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
#         Focus on key findings, recommendations, and next steps.
#         """
#         context = {"specialty": "report"}
#         result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Report summary error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/chatbot/session/clear")
# async def clear_session(request: ClearSessionRequest):
#     """Clear chatbot session history."""
#     try:
#         session_id = request.session_id or "default_session"
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
#             conn.commit()
#         conn.close()
#         return {"message": "Session cleared successfully", "session_id": session_id}
#     except Exception as e:
#         logger.error(f"Clear session error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to clear session")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": "2.1.0",
#         "agents_available": True,
#         "thinking_enabled": True
#     }

# @app.get("/api/chatbot/sessions")
# async def get_sessions():
#     """Get active session count (for monitoring)."""
#     try:
#         conn = get_db()
#         with conn.cursor() as cur:
#             cur.execute("SELECT COUNT(*) FROM chat_sessions")
#             active_sessions = cur.fetchone()[0]
#             cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
#             total_messages_result = cur.fetchone()[0]
#             total_messages = total_messages_result if total_messages_result is not None else 0
#         conn.close()
#         return {
#             "active_sessions": active_sessions,
#             "total_messages": total_messages
#         }
#     except Exception as e:
#         logger.error(f"Get sessions error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve session data")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# =======================================remove commnets===========================================



import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
import logging
from contextlib import asynccontextmanager
import uuid
import httpx
import re
import pymysql
import google.generativeai as genai 
import aiohttp



# In main.py and cardiology_ai.py
from utils import search_similar_cases, fallback_text_search
# Import OpenAI Agents SDK components (assumed to be available)
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# Import agent creation functions from medicura_agents and specialist_agents
from medicura_agents.symptom_analyzer_agent import create_symptom_analyzer_agent
from medicura_agents.drug_interaction_agent import create_drug_interaction_agent
from medicura_agents.general_health_agent import create_general_health_agent
from medicura_agents.medical_term_agent import create_medical_term_agent
from medicura_agents.report_analyzer_agent import create_report_analyzer_agent
from medicura_agents.about_agent import create_about_agent


from specialist_agents.cardiology_ai import create_cardiology_agent
from specialist_agents.dermatology_ai import create_dermatology_agent
from specialist_agents.neurology_ai import create_neurology_agent
from specialist_agents.pulmonology_ai import create_pulmonology_agent
from specialist_agents.ophthalmology_ai import create_ophthalmology_agent
from specialist_agents.dental_ai import create_dental_agent
from specialist_agents.allergy_immunology_ai import create_allergy_immunology_agent
from specialist_agents.pediatrics_ai import create_pediatrics_agent
from specialist_agents.orthopedics_ai import create_orthopedics_agent
from specialist_agents.mental_health_ai import create_mental_health_agent
from specialist_agents.endocrinology_ai import create_endocrinology_agent
from specialist_agents.gastroenterology_ai import create_gastroenterology_agent
from specialist_agents.radiology_ai import create_radiology_agent
from specialist_agents.infectious_disease_ai import create_infectious_disease_agent
from specialist_agents.vaccination_advisor_ai import create_vaccination_advisor_agent
from specialist_agents.drug_interaction_agent import create_drug_interaction_agent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


# TiDB Configuration with minimal SSL
DB_CONFIG = {
    "host": os.getenv("TIDB_HOST", "gateway01.us-west-2.prod.aws.tidbcloud.com"),
    "port": 4000,
    "user": os.getenv("TIDB_USERNAME", "34oY1b3G6arXWAM.root"),
    "password": os.getenv("TIDB_PASSWORD", "M9iWYjgizxiiT1qh"),
    "database": os.getenv("TIDB_DATABASE", "test"),
    "charset": "utf8mb4",
    "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
}

def get_db():
    """Establish a connection to TiDB."""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except pymysql.err.OperationalError as e:
        logger.error(f"Failed to connect to TiDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("Starting Medicura-AI Health Assistant")
    logger.info("Connecting to TiDB...")
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(100) PRIMARY KEY,
                    history JSON NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS specialist_vectors (
                    id VARCHAR(100) PRIMARY KEY,
                    specialty VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(768) NOT NULL,  -- CHANGED FROM JSON TO VECTOR(768)
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        conn.close()
        logger.info("TiDB connected and tables ready.")
        yield
    except Exception as e:
        logger.error(f"Lifespan error: {str(e)}")
        raise HTTPException(status_code=500, detail="Application startup failed")
    finally:
        logger.info("Shutting down Medicura-AI Health Assistant")

app = FastAPI(
    title="Medicura-AI Health Assistant",
    description="AI-powered health assistant for symptom analysis and medical queries",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://hg-medicura-ai.vercel.app",
    "https://*.vercel.app",
    "https://*.railway.app",
]

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variable Validation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY environment variable is required")


# AI Agent Initialization
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=httpx.AsyncClient(timeout=60.0)
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

model_settings = ModelSettings(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    model_settings=model_settings,
    tracing_disabled=True,
)

# Initialize Core Agents (Specialist agents ko temporarily comment out)
symptom_analyzer_agent = create_symptom_analyzer_agent(model)
drug_interaction_agent = create_drug_interaction_agent(model)
general_health_agent = create_general_health_agent(model)
medical_term_agent = create_medical_term_agent(model)
report_analyzer_agent = create_report_analyzer_agent(model)
about_agent = create_about_agent(model)


# Specialist agents
cardiology_agent = create_cardiology_agent(model)
dermatology_agent = create_dermatology_agent(model)
neurology_agent = create_neurology_agent(model)
pulmonology_agent = create_pulmonology_agent(model)
ophthalmology_agent = create_ophthalmology_agent(model)
dental_agent = create_dental_agent(model)
allergy_immunology_agent = create_allergy_immunology_agent(model)
pediatrics_agent = create_pediatrics_agent(model)
orthopedics_agent = create_orthopedics_agent(model)
mental_health_agent = create_mental_health_agent(model)
endocrinology_agent = create_endocrinology_agent(model)
gastroenterology_agent = create_gastroenterology_agent(model)
radiology_agent = create_radiology_agent(model)
infectious_disease_agent = create_infectious_disease_agent(model)
vaccination_advisor_agent = create_vaccination_advisor_agent(model)
drug_interaction_agent = create_drug_interaction_agent(model)


# Configure the Gemini client (add this near your other config code, e.g., after loading env vars)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_embedding(text: str) -> List[float]:
    """Generate a real embedding vector using the Gemini embedding model."""
    try:
        # Call the Gemini Embedding API
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document" # Or "retrieval_query", "classification", etc.
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        # Fallback to avoid breaking the application, but log the error heavily.
        return [0.0] * 768
    





# === FDA DRUG API INTEGRATION === #
async def fetch_fda_drug_info(drug_name: str):
    """Fetch drug information from FDA API"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name.lower()}&limit=1"
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        return data['results'][0]
                return None
    except Exception as e:
        logger.error(f"FDA API error for {drug_name}: {str(e)}")
        return None
# === END FDA INTEGRATION === #








def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from agent response with multiple fallback methods."""
    try:
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        return {
            "summary": response,
            "detailed_analysis": "Detailed analysis based on your query",
            "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
            "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
            "type": "general"
        }
    except Exception as e:
        logger.warning(f"JSON extraction failed: {str(e)}")
        return None

async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run agent with enhanced thinking and robust error handling."""
    try:
        specialty = context.get("specialty", "general") if context else "general"
        
        # For drug-related queries, provide more specific context
        if specialty == "drug":
            thinking_prompt = f"""
            USER QUERY: {prompt}
            CONTEXT: This is a drug-related query. Please provide information about usage, dosage, precautions, and interactions.
            
            PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
            DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
            """

        elif specialty == "symptom":
            thinking_prompt = f"""
            USER QUERY: {prompt}
            CONTEXT: This is a symptom analysis query. Provide comprehensive information about 
            possible causes, self-care measures, when to seek help, and warning signs.
            
            RESPONSE FORMAT: Provide a comprehensive JSON response with detailed fields.
            """

        else:
            thinking_prompt = f"""
            USER QUERY: {prompt}
            CONTEXT: {json.dumps(context) if context else 'No additional context'}
            
            PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
            DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
            """
        
        result = await Runner.run(agent, thinking_prompt, run_config=config)
        logger.info(f"Raw agent response: {result.final_output[:200]}...")

        
        parsed_response = extract_json_from_response(result.final_output)
        
        if parsed_response:
            parsed_response.update({
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "thinking_applied": True
            })
            return parsed_response
        else:
            return create_intelligent_response(result.final_output, prompt, specialty)
        
    except Exception as e:
        logger.error(f"Agent error: {str(e)}")
        # Create a fallback response based on the query
        if "headache" in prompt.lower() and "panadol" in prompt.lower():
            return {
                "summary": "Panadol (paracetamol) can generally be taken for headaches",
                "detailed_analysis": "Panadol (paracetamol) is commonly used for headache relief. The typical adult dosage is 500-1000mg every 4-6 hours as needed, not exceeding 4000mg in 24 hours. Make sure you don't have any contraindications like liver disease.",
                "recommendations": [
                    "Follow dosage instructions on packaging",
                    "Don't exceed maximum daily dose",
                    "Consult doctor if headache persists beyond 3 days"
                ],
                "when_to_seek_help": [
                    "If headache is severe or sudden",
                    "If accompanied by fever, stiff neck, or vision changes",
                    "If headache persists despite medication"
                ],
                "disclaimer": "This is general information. Consult healthcare professionals for personalized advice.",
                "type": "drug",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        return create_intelligent_response(f"Analysis of: {prompt}")
    

def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
    """Create a well-structured response from text."""
    return {
        "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
        "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
        "recommendations": [
            "Consult with a healthcare provider",
            "Provide complete medical history for assessment",
            "Follow evidence-based medical guidance"
        ],
        "when_to_seek_help": [
            "Immediately for severe or emergency symptoms",
            "Within 24-48 hours for persistent concerns",
            "Routinely for preventive care"
        ],
        "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
        "type": "general",
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "thinking_applied": True
    }

# -----
def create_structured_response_from_text(text: str, original_query: str, specialty: str) -> Dict[str, Any]:
    """Create a structured response when JSON parsing fails."""
    base_response = {
        "summary": text[:150] + "..." if len(text) > 150 else text,
        "detailed_analysis": text,
        "timestamp": datetime.now().isoformat(),
        "success": True,
        "thinking_applied": True
    }
    
    # Add specialty-specific fields
    if specialty == "drug":
        base_response.update({
            "type": "drug",
            "recommendations": ["Follow dosage instructions", "Consult doctor if unsure", "Read medication leaflet"],
            "disclaimer": "This is general information. Consult healthcare professionals for personalized advice."
        })
    elif specialty == "symptom":
        base_response.update({
            "type": "symptom",
            "when_to_seek_help": ["If symptoms persist", "If severe pain", "If symptoms worsen"],
            "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice."
        })
    
    return base_response

async def run_multi_agent_workflow(prompt: str, context: Dict = None):
    """Chain multiple agents for comprehensive analysis."""
    # Symptom → Drug → General Health chain
    symptom_result = await Runner.run(symptom_analyzer_agent, prompt, run_config=config)
    drug_result = await Runner.run(drug_interaction_agent, f"Symptoms: {prompt}\nAnalysis: {symptom_result.final_output}", run_config=config)
    health_result = await Runner.run(general_health_agent, f"Symptoms: {prompt}\nDrug Analysis: {drug_result.final_output}", run_config=config)
    
    return {
        "symptom_analysis": extract_json_from_response(symptom_result.final_output),
        "drug_analysis": extract_json_from_response(drug_result.final_output),
        "health_analysis": extract_json_from_response(health_result.final_output),
        "multi_agent_workflow": True
    }

def load_history(session_id: str) -> List[dict]:
    """Load chat history from TiDB."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
            result = cur.fetchone()
        conn.close()
        return json.loads(result[0]) if result else []
    except Exception as e:
        logger.error(f"Failed to load history: {str(e)}")
        return []

def save_history(session_id: str, history: List[dict]):
    """Save chat history to TiDB."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (session_id, history)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
            """, (session_id, json.dumps(history), json.dumps(history)))
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save chat history")

# Pydantic Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, max_length=100)
    context: Optional[dict] = None

class DrugInteractionInput(BaseModel):
    medications: List[str] = Field(..., min_items=1, max_items=10)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=20)
    existing_conditions: Optional[List[str]] = Field(None, max_items=20)
    other_medications: Optional[List[str]] = Field(None, max_items=20)

class MedicalTermInput(BaseModel):
    term: str = Field(..., min_length=1, max_length=100)
    language: Optional[str] = Field("en", max_length=10)

class ReportTextInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    language: Optional[str] = Field("en", max_length=10)

class ClearSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, max_length=100)


#-----------------------FDA------------------------- #

@app.get("/api/test/fda-drug")
async def test_fda_drug(drug_name: str = "aspirin"):
    """Test FDA drug API integration."""
    try:
        result = await fetch_fda_drug_info(drug_name)
        return {
            "drug_name": drug_name,
            "fda_data_available": result is not None,
            "data": result if result else "No FDA data found",
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}
    
#-----------------------FDA------------------------- #


@app.post("/api/chatbot")
async def chatbot(request: ChatRequest):
    """Main chatbot endpoint with intelligent thinking and specialty support."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        history = load_history(session_id)

        # Select appropriate agent based on specialty and keywords
        query_lower = request.message.lower()
        logger.info(f"Received query: {query_lower}")
       
        specialty_map = {
            "symptom": ["symptom", "pain", "fever", "headache", "nausea", "ache", "hurt"],
            "drug": ["drug", "medication", "pill", "dose", "interaction", "side effect", "ibuprofen", "panadol", "paracetamol"],
            "medical_term": ["what is", "explain", "define", "meaning of"],
            "report": ["report", "result", "test", "lab", "x-ray", "summary"],
            "about": ["creator", "author", "hadiqa", "gohar", "medicura about", "who made"],
            "cardiology": ["heart", "cardio", "chest pain", "palpitations"],
            "dermatology": ["skin", "rash", "eczema", "psoriasis"],
            "neurology": ["brain", "migraine", "seizure", "numbness"],
            "pulmonology": ["lung", "cough", "asthma", "bronchitis"],
            "ophthalmology": ["eye", "vision", "blurred vision", "cataract"],
            "dental": ["tooth", "dentist", "toothache", "gum"],
            "allergy_immunology": ["allergy", "sneeze", "immunology", "pollen"],
            "pediatrics": ["child", "baby", "infant", "pediatric"],
            "orthopedics": ["bone", "joint", "fracture", "arthritis"],
            "mental_health": ["mental", "stress", "depression", "anxiety"],
            "endocrinology": ["hormone", "thyroid", "diabetes", "endocrine"],
            "gastroenterology": ["stomach", "abdomen", "gastritis", "ulcer"],
            "radiology": ["x-ray", "mri", "ct scan", "radiology"],
            "infectious_disease": ["flu", "infection", "virus", "bacteria"],
            "vaccination_advisor": ["vaccine", "immunization", "vaccination"]
        }

        selected_specialty = "general"
        selected_agent = general_health_agent

        for specialty, keywords in specialty_map.items():
            if any(keyword in query_lower for keyword in keywords):
                selected_specialty = specialty
                logger.info(f"Selected specialty: {specialty}")
                
                # Map specialties to agents
                agent_mapping = {
                    "symptom": symptom_analyzer_agent,
                    "drug": drug_interaction_agent,
                    "medical_term": medical_term_agent,
                    "report": report_analyzer_agent,
                    "about": about_agent,
                    "cardiology": cardiology_agent,
                    "dermatology": dermatology_agent,
                    "neurology": neurology_agent,
                    "pulmonology": pulmonology_agent,
                    "ophthalmology": ophthalmology_agent,
                    "dental": dental_agent,
                    "allergy_immunology": allergy_immunology_agent,
                    "pediatrics": pediatrics_agent,
                    "orthopedics": orthopedics_agent,
                    "mental_health": mental_health_agent,
                    "endocrinology": endocrinology_agent,
                    "gastroenterology": gastroenterology_agent,
                    "radiology": radiology_agent,
                    "infectious_disease": infectious_disease_agent,
                    "vaccination_advisor": vaccination_advisor_agent
                }
                
                selected_agent = agent_mapping.get(specialty, general_health_agent)
                break

        logger.info(f"Final selected agent: {selected_specialty}")

        # Run agent with thinking mode
        context = {"specialty": selected_specialty}
        result = await run_agent_with_thinking(selected_agent, request.message, context)

        # Update chat history
        history.extend([
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": json.dumps(result), "timestamp": datetime.now().isoformat()}
        ])
        history = history[-20:]  # Keep last 20 messages
        save_history(session_id, history)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
        )

@app.get("/api/test/vector-search")
async def test_vector_search(query: str = "chest pain", specialty: str = "cardiology"):
    """Test endpoint for vector search functionality."""
    try:
        results = search_similar_cases(query, specialty)
        return {
            "query": query,
            "specialty": specialty, 
            "results": results,
            "vector_search_working": True
        }
    except Exception as e:
        return {"error": str(e), "vector_search_working": False}    

# @app.post("/api/health/drug-interactions")
# async def check_drug_interactions(input_data: DrugInteractionInput):
#     """Check drug interactions with thorough analysis."""
#     try:
#         if not input_data.medications or len(input_data.medications) == 0:
#             raise HTTPException(status_code=400, detail="At least one medication is required")
        
#             # Fetch FDA info for the first medication (as example)
#         fda_data = None
#         if input_data.medications:
#             fda_data = await fetch_fda_drug_info(input_data.medications[0])
        
    
#         context = {
#             "medications": input_data.medications,
#             "age": input_data.age,
#             "gender": input_data.gender,
#             "existing_conditions": input_data.existing_conditions,
#             "other_medications": input_data.other_medications,
#             "specialty": "drug"
#         }
        
#         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
#         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Drug interaction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")


@app.post("/api/health/drug-interactions")
async def check_drug_interactions(input_data: DrugInteractionInput):
    """Check drug interactions with thorough analysis."""
    try:
        if not input_data.medications or len(input_data.medications) == 0:
            raise HTTPException(status_code=400, detail="At least one medication is required")
        
        # Fetch FDA info for the first medication (as example)
        fda_data = None
        if input_data.medications:
            fda_data = await fetch_fda_drug_info(input_data.medications[0])
        
        context = {
            "medications": input_data.medications,
            "age": input_data.age,
            "gender": input_data.gender,
            "existing_conditions": input_data.existing_conditions,
            "other_medications": input_data.other_medications,
            "specialty": "drug",
            "fda_data": fda_data  # Add FDA data to context
        }
        
        prompt = f"Check interactions for: {', '.join(input_data.medications)}"
        result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
        return result
        
    except Exception as e:
        logger.error(f"Drug interaction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")


@app.post("/api/health/medical-term")
async def explain_medical_term(input_data: MedicalTermInput):
    """Explain medical terms with clarity."""
    try:
        if not input_data.term:
            raise HTTPException(status_code=400, detail="Medical term is required")
        
        prompt = f"Explain the medical term: {input_data.term}"
        if input_data.language and input_data.language != "en":
            prompt += f" in {input_data.language} language"
        
        context = {"specialty": "medical_term"}
        result = await run_agent_with_thinking(medical_term_agent, prompt, context)
        return result
        
    except Exception as e:
        logger.error(f"Medical term error: {str(e)}")
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")

@app.post("/api/health/report-summarize")
async def summarize_medical_report(input_data: ReportTextInput):
    """Summarize medical reports with intelligent analysis."""
    try:
        if not input_data.text:
            raise HTTPException(status_code=400, detail="Report text is required")
        
        prompt = f"""
        Analyze and summarize this medical report:

        {input_data.text}

        Please provide the summary in {input_data.language if input_data.language else 'English'} language.
        Focus on key findings, recommendations, and next steps.
        """
        context = {"specialty": "report"}
        result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
        return result
        
    except Exception as e:
        logger.error(f"Report summary error: {str(e)}")
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")

@app.post("/api/chatbot/session/clear")
async def clear_session(request: ClearSessionRequest):
    """Clear chatbot session history."""
    try:
        session_id = request.session_id or "default_session"
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
            conn.commit()
        conn.close()
        return {"message": "Session cleared successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Clear session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "agents_available": True,
        "thinking_enabled": True
    }

@app.get("/api/chatbot/sessions")
async def get_sessions():
    """Get active session count (for monitoring)."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_sessions")
            active_sessions = cur.fetchone()[0]
            cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
            total_messages_result = cur.fetchone()[0]
            total_messages = total_messages_result if total_messages_result is not None else 0
        conn.close()
        return {
            "active_sessions": active_sessions,
            "total_messages": total_messages
        }
    except Exception as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




