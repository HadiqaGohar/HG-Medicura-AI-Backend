

import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pdfplumber




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

# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "HG-Medicura-AI Backend is running 🚀"}

# CORS Configuration - SPECIFIC domains without wildcards
origins = [
    "https://hg-medicura-ai.vercel.app",  # Your Vercel frontend
    "http://localhost:3000",
    "http://localhost:3001",
    "https://hg-medicura-ai-backend-production.up.railway.app",  # Your Railway backend
]

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

# ......................Add new one................. #

class SymptomAnalyzerRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1, max_items=20)
    duration: Optional[str] = Field("not specified", max_length=100)
    severity: Optional[str] = Field("not specified", max_length=100)

class DrugInteractionRequest(BaseModel):
    medications: List[str] = Field(..., min_items=1, max_items=20)
    age: Optional[int] = None
    gender: Optional[str] = None
    existing_conditions: Optional[List[str]] = []
    other_medications: Optional[List[str]] = []
    
class MedicalTermRequest(BaseModel):
    term: str = Field(..., min_length=1)
    language: Optional[str] = "en"    

# class ReportSummaryRequest(BaseModel):
#     text: str = Field(..., min_length=1)
#     language: Optional[str] = "en"

class ReportSummaryResponse(BaseModel):
    summary: str
    detailed_analysis: str
    key_findings: List[str]
    recommendations: List[str]
    next_steps: List[str]
    disclaimer: str
    type: str
    error: Optional[str] = None



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
#-----------------------Medicura Agents------------------------- #


@app.post("/api/health/symptom-analyzer")
async def symptom_analyzer(request: SymptomAnalyzerRequest):
    """Analyze symptoms using the symptom_analyzer_agent."""
    try:
        symptoms_str = ", ".join(request.symptoms)
        prompt = f"""
        Analyze the following symptoms:
        Symptoms: {symptoms_str}
        Duration: {request.duration}
        Severity: {request.severity}
        """
        context = {"specialty": "symptom"}
        result = await run_agent_with_thinking(symptom_analyzer_agent, prompt, context)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Symptom analyzer error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze symptoms")

@app.post("/api/health/drug-interactions")
async def drug_interactions(request: DrugInteractionRequest):
    try:
        medications_str = ", ".join(request.medications)
        prompt = f"Analyze drug interactions for: Medications: {medications_str}"
        if request.age:
            prompt += f", Age: {request.age}"
        if request.gender:
            prompt += f", Gender: {request.gender}"
        if request.existing_conditions:
            prompt += f", Conditions: {', '.join(request.existing_conditions)}"
        if request.other_medications:
            prompt += f", Other Medications: {', '.join(request.other_medications)}"
        prompt = prompt.strip()
        context = {"specialty": "drug_interaction"}

        # Call agent
        result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
        
        # Log response safely
        result_str = str(result) if not isinstance(result, str) else result
        logger.info(f"Drug interaction raw response: {result_str[:200]}...")
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Drug interaction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check drug interactions")

@app.post("/api/health/medical-term")
async def medical_term(request: MedicalTermRequest):
    try:
        prompt = (
            f"Explain the medical term '{request.term}' in {request.language}. "
            f"Provide a JSON response with the following structure: "
            f"{{'term': the term, 'pronunciation': phonetic spelling, "
            f"'summary': brief definition, 'detailed_analysis': concise explanation, "
            f"'key_points': array of bullet points, 'related_terms': array of related terms, "
            f"'recommendations': actionable advice or 'None', "
            f"'disclaimer': disclaimer text, 'type': 'medical_term'}}."
        )
        context = {"specialty": "medical_term"}

        # Call agent
        result = await run_agent_with_thinking(medical_term_agent, prompt, context)
        
        # Log response safely
        result_str = str(result) if not isinstance(result, str) else result
        logger.info(f"Medical term raw response: {result_str[:200]}...")
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Medical term error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to explain medical term")


async def extract_pdf_text(file: UploadFile) -> str:
    try:
        # Validate file type
        if file.content_type != 'application/pdf':
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract text from PDF
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file.filename}")
                raise HTTPException(status_code=400, detail="No readable text found in PDF")
        logger.info(f"Extracted text from PDF: {file.filename} ({len(text)} characters)")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

@app.post("/api/health/report-summarize", response_model=ReportSummaryResponse)
async def report_summarize(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form("en")
):
    try:
        report_text = text or ""
        if file:
            report_text = await extract_pdf_text(file)

        if not report_text.strip():
            logger.error("No valid report text provided")
            raise HTTPException(status_code=400, detail="Report text or valid PDF file is required")

        prompt = (
            f"Summarize the following medical report text in {language}: {report_text}. "
            f"Provide a JSON response with the following structure: "
            f"{{'summary': brief summary, 'detailed_analysis': detailed explanation, "
            f"'key_findings': array of findings, 'recommendations': array of recommendations, "
            f"'next_steps': array of next steps, 'disclaimer': disclaimer text, "
            f"'type': report type (e.g., 'Lab Results', 'Imaging Reports', 'Doctor's Notes', 'Discharge Summaries')}}."
        )
        context = {"specialty": "report_analyzer"}

        # Call agent
        result = await run_agent_with_thinking(report_analyzer_agent, prompt, context)
        
        # Validate response
        if not isinstance(result, dict):
            logger.error(f"Agent returned non-dict response: {type(result)}")
            raise HTTPException(status_code=500, detail="Invalid agent response format")

        # Ensure all required fields
        formatted_result = {
            "summary": result.get("summary", "No summary available."),
            "detailed_analysis": result.get("detailed_analysis", result.get("summary", "No detailed analysis available.")),
            "key_findings": result.get("key_findings", []),
            "recommendations": result.get("recommendations", []),
            "next_steps": result.get("next_steps", []),
            "disclaimer": result.get("disclaimer", f"This information is educational only and not medical advice in {language}."),
            "type": result.get("type", "Unknown"),
            "error": result.get("error", None)
        }

        logger.info(f"Report summary response: {str(formatted_result)[:200]}...")
        return JSONResponse(content=formatted_result)
    except HTTPException as e:
        logger.error(f"HTTP error in report summarize: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in report summarize: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to summarize report",
                "summary": "",
                "detailed_analysis": "Unable to summarize the report. Please try again or consult a healthcare provider.",
                "key_findings": [],
                "recommendations": [],
                "next_steps": [],
                "disclaimer": f"This information is educational only and not medical advice in {language}.",
                "type": "Unknown"
            }
        )
# -----------------End new one .....................#


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
        # ad new ......
@app.post("/api/chatbot/session/clear")
async def clear_session(request: dict):
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    # Implement session clearing logic (e.g., clear database entries for session_id)
    return {"message": f"Session {session_id} cleared"}        

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

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# import os

# PORT = int(os.environ.get("PORT", 8080))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")



# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )
