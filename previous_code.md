# # # # # # # import os
# # # # # # # import io
# # # # # # # import json
# # # # # # # from datetime import datetime
# # # # # # # from fastapi import FastAPI, HTTPException, Request, UploadFile, File
# # # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # # from pydantic import BaseModel
# # # # # # # from dotenv import load_dotenv
# # # # # # # import pdfplumber
# # # # # # # import mammoth
# # # # # # # from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
# # # # # # # from typing import Optional, Dict


# # # # # # # # Load environment variables from .env file
# # # # # # # load_dotenv()

# # # # # # # app = FastAPI()

# # # # # # # # --- CORS Configuration ---
# # # # # # # app.add_middleware(
# # # # # # #     CORSMiddleware,
# # # # # # #     # allow_origins=["http://localhost:3000", "https://hg-ai-resume-builder.vercel.app/"],  # Replace with your Vercel URL
# # # # # # #     allow_origins=["*"],  # For development, allow all origins. Change in production.
# # # # # # #     allow_credentials=True,
# # # # # # #     allow_methods=["*"],
# # # # # # #     allow_headers=["*"],
# # # # # # # )

# # # # # # # # --- Environment Variable Loading ---
# # # # # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # # # # # # # --- AI Agent Initialization (Gemini via OpenAI SDK compatibility) ---
# # # # # # # external_client = AsyncOpenAI(
# # # # # # #     api_key=GEMINI_API_KEY,
# # # # # # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # # # # # )

# # # # # # # model = OpenAIChatCompletionsModel(
# # # # # # #     openai_client=external_client,
# # # # # # #     model="gemini-2.0-flash",
# # # # # # # )

# # # # # # # config = RunConfig(
# # # # # # #     model=model,
# # # # # # #     model_provider=external_client,
# # # # # # #     tracing_disabled=True,
# # # # # # # )

# # # # # # # # Pydantic models for request body
# # # # # # # class ResumeInput(BaseModel):
# # # # # # #     education: list[str]
# # # # # # #     skills: list[str]

# # # # # # # class ResumeData(BaseModel):
# # # # # # #     name: str = ""
# # # # # # #     tag: str = ""
# # # # # # #     email: str = ""
# # # # # # #     location: str = ""
# # # # # # #     number: str = ""
# # # # # # #     phone: str = ""
# # # # # # #     summary: str = ""
# # # # # # #     websites: list[str] = []
# # # # # # #     website: str = ""
# # # # # # #     linkedin: str = ""
# # # # # # #     github: str = ""
# # # # # # #     skills: list[str] = []
# # # # # # #     education: list[str] = []
# # # # # # #     experience: list[str] = []
# # # # # # #     student: list[str] = []
# # # # # # #     courses: list[str] = []
# # # # # # #     internships: list[str] = []
# # # # # # #     extracurriculars: list[str] = []
# # # # # # #     hobbies: list[str] = []
# # # # # # #     references: list[str] = []
# # # # # # #     languages: list[str] = []
# # # # # # #     awards: list[str] = []
# # # # # # #     extra: list[str] = []
# # # # # # #     certifications: list = []
# # # # # # #     projects: list = []
# # # # # # #     headerColor: str = "#a3e4db"
# # # # # # #     nameFontStyle: str = "regular"
# # # # # # #     nameFontSize: int = 18
# # # # # # #     tagFontStyle: str = "regular"
# # # # # # #     tagFontSize: int = 14
# # # # # # #     summaryFontStyle: str = "regular"
# # # # # # #     summaryFontSize: int = 12
# # # # # # #     image: str = ""
# # # # # # #     profileImage: str = ""

# # # # # # # class JobOptimizationInput(BaseModel):
# # # # # # #     job_description: str
# # # # # # #     resume_data: ResumeData

# # # # # # # class SkillSuggestionInput(BaseModel):
# # # # # # #     profession: str
# # # # # # #     current_skills: list[str] = []




# # # # # # # @app.post("/api/resume")
# # # # # # # async def generate_resume_summary(input_data: ResumeInput):
# # # # # # #     try:
# # # # # # #         # Convert lists to comma-separated strings for the prompt
# # # # # # #         education_str = ", ".join(input_data.education)
# # # # # # #         skills_str = ", ".join(input_data.skills)
# # # # # # #         # extra_str = ", ".join(input_data.extra)
# # # # # # #         # student_str = ", ".join(input_data.student)
# # # # # # #         # experiene_str = ", ".join(input_data.experience)
# # # # # # #         # language_str = ", ".join(input_data.language)
# # # # # # #         # award_str = ", ".join(input_data.award)

# # # # # # #         # Call Gemini model via OpenAI SDK compatibility
# # # # # # #         response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[
# # # # # # #                 {
# # # # # # #                     "role": "user",
# # # # # # #                     "content": f"Generate a professional resume summary for a candidate with education: {education_str} and skills: {skills_str}. Keep it concise, professional, and ATS-friendly. Limit to 3-4 sentences."
# # # # # # #                 }
# # # # # # #             ]
# # # # # # #         )

# # # # # # #         summary = response.choices[0].message.content or ""
# # # # # # #         return {"summary": summary}

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error generating resume summary: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# # # # # # # @app.post("/api/resume/extract")
# # # # # # # async def extract_resume_data(file: UploadFile = File(...)):
# # # # # # #     """Extract structured data from uploaded resume"""
# # # # # # #     if not file.filename:
# # # # # # #         raise HTTPException(status_code=400, detail="No file provided")
    
# # # # # # #     extracted_text = ""
    
# # # # # # #     try:
# # # # # # #         content = await file.read()
        
# # # # # # #         # Extract text based on file type
# # # # # # #         if file.filename.lower().endswith(".pdf"):
# # # # # # #             import io
# # # # # # #             pdf_file = io.BytesIO(content)
# # # # # # #             with pdfplumber.open(pdf_file) as pdf:
# # # # # # #                 for page in pdf.pages:
# # # # # # #                     page_text = page.extract_text() or ""
# # # # # # #                     extracted_text += page_text + "\n"
                    
# # # # # # #         elif file.filename.lower().endswith(".docx"):
# # # # # # #             import io
# # # # # # #             import mammoth
# # # # # # #             docx_file = io.BytesIO(content)
# # # # # # #             result = mammoth.extract_raw_text(docx_file)
# # # # # # #             extracted_text = result.value
# # # # # # #         else:
# # # # # # #             raise HTTPException(
# # # # # # #                 status_code=400,
# # # # # # #                 detail="Unsupported file format. Only PDF and DOCX files are supported."
# # # # # # #             )
        
# # # # # # #         if not extracted_text.strip():
# # # # # # #             raise HTTPException(
# # # # # # #                 status_code=400,
# # # # # # #                 detail="No text could be extracted from the file."
# # # # # # #             )
        
# # # # # # #         # Use Gemini to parse extracted text
# # # # # # #         response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[
# # # # # # #                 {
# # # # # # #                     "role": "user",
# # # # # # #                     "content": f"""Parse the following resume text into structured JSON data. Extract these fields:
# # # # # # #                     - name (string): Full name
# # # # # # #                     - tag (string): Professional title or role  
# # # # # # #                     - email (string): Email address
# # # # # # #                     - location (string): City, State or address
# # # # # # #                     - number (string): Phone number
# # # # # # #                     - summary (string): Professional summary
# # # # # # #                     - websites (array): URLs like LinkedIn, portfolio
# # # # # # #                     - skills (array): Technical and soft skills
# # # # # # #                     - education (array): Degrees, schools, years
# # # # # # #                     - experience (array): Job titles, companies
# # # # # # #                     - student (array): Student status
# # # # # # #                     - courses (array): Relevant coursework
# # # # # # #                     - internships (array): Internship experiences
# # # # # # #                     - extracurriculars (array): Activities, volunteer work
# # # # # # #                     - hobbies (array): Personal interests
# # # # # # #                     - references (array): Professional references
# # # # # # #                     - languages (array): Spoken languages

# # # # # # #                     Return ONLY valid JSON without markdown formatting.
# # # # # # #                     Use empty string "" for missing fields and empty array [] for missing lists.

# # # # # # #                     Resume text:
# # # # # # #                     {extracted_text}
# # # # # # #                     """
# # # # # # #                 }
# # # # # # #             ],
# # # # # # #             max_tokens=1500,
# # # # # # #             temperature=0.3
# # # # # # #         )
        
# # # # # # #         result = response.choices[0].message.content or "{}"
        
# # # # # # #         # Clean the response
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             structured_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             # Fallback structure if JSON parsing fails
# # # # # # #             structured_data = {
# # # # # # #                 "name": "",
# # # # # # #                 "tag": "",
# # # # # # #                 "email": "",
# # # # # # #                 "location": "",
# # # # # # #                 "number": "",
# # # # # # #                 "summary": "",
# # # # # # #                 "websites": [],
# # # # # # #                 "skills": [],
# # # # # # #                 "education": [],
# # # # # # #                 "experience": [],
# # # # # # #                 "student": [],
# # # # # # #                 "courses": [],
# # # # # # #                 "internships": [],
# # # # # # #                 "extracurriculars": [],
# # # # # # #                 "hobbies": [],
# # # # # # #                 "references": [],
# # # # # # #                 "languages": []
# # # # # # #             }
        
# # # # # # #         return structured_data
        
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error extracting resume data: {str(e)}")
# # # # # # #         raise HTTPException(
# # # # # # #             status_code=500, 
# # # # # # #             detail=f"Failed to extract resume data: {str(e)}"
# # # # # # #         )

# # # # # # # @app.get("/health")
# # # # # # # async def health_check():
# # # # # # #     """Health check endpoint"""
# # # # # # #     try:
# # # # # # #         # Test Gemini connection
# # # # # # #         test_response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[{"role": "user", "content": "Hello"}],
# # # # # # #             max_tokens=10
# # # # # # #         )
# # # # # # #         gemini_status = "connected"
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Gemini connection failed: {e}")
# # # # # # #         gemini_status = "disconnected"
    
# # # # # # #     return {
# # # # # # #         "api_status": "healthy",
# # # # # # #         "gemini_status": gemini_status,
# # # # # # #         "timestamp": datetime.now().isoformat()
# # # # # # #     }

# # # # # # # @app.post("/api/resume/edit")
# # # # # # # async def edit_resume_data(resume_data: ResumeData):
# # # # # # #     """Save/edit resume data"""
# # # # # # #     try:
# # # # # # #         # For now, just return the data as confirmation
# # # # # # #         # In a real app, you'd save this to a database
# # # # # # #         return resume_data.dict()
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error editing resume data: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to edit resume data: {str(e)}")

# # # # # # # @app.post("/api/resume/optimize")
# # # # # # # async def optimize_resume(input_data: JobOptimizationInput):
# # # # # # #     """Optimize resume for specific job description"""
# # # # # # #     try:
# # # # # # #         job_desc = input_data.job_description
# # # # # # #         resume = input_data.resume_data
        
# # # # # # #         # Create optimization prompt
# # # # # # #         prompt = f"""
# # # # # # #         Analyze this job description and optimize the resume accordingly:
        
# # # # # # #         JOB DESCRIPTION:
# # # # # # #         {job_desc}
        
# # # # # # #         CURRENT RESUME DATA:
# # # # # # #         Name: {resume.name}
# # # # # # #         Title: {resume.tag}
# # # # # # #         Summary: {resume.summary}
# # # # # # #         Skills: {', '.join(resume.skills)}
# # # # # # #         Experience: {', '.join(resume.experience)}
# # # # # # #         Education: {', '.join(resume.education)}
        
# # # # # # #         Please provide:
# # # # # # #         1. An optimized professional summary that matches the job requirements
# # # # # # #         2. 5-8 additional skills that would be relevant for this job
# # # # # # #         3. Key keywords from the job description that match the resume
# # # # # # #         4. 3-5 specific improvement suggestions
        
# # # # # # #         Return the response in this exact JSON format:
# # # # # # #         {{
# # # # # # #             "optimized_summary": "...",
# # # # # # #             "suggested_skills": ["skill1", "skill2", ...],
# # # # # # #             "keyword_matches": ["keyword1", "keyword2", ...],
# # # # # # #             "improvement_suggestions": ["suggestion1", "suggestion2", ...]
# # # # # # #         }}
# # # # # # #         """
        
# # # # # # #         response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[{"role": "user", "content": prompt}],
# # # # # # #             max_tokens=1500,
# # # # # # #             temperature=0.3
# # # # # # #         )
        
# # # # # # #         result = response.choices[0].message.content or "{}"
        
# # # # # # #         # Clean the response
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             optimization_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             # Fallback response
# # # # # # #             optimization_data = {
# # # # # # #                 "optimized_summary": resume.summary,
# # # # # # #                 "suggested_skills": [],
# # # # # # #                 "keyword_matches": [],
# # # # # # #                 "improvement_suggestions": ["Unable to generate specific suggestions at this time."]
# # # # # # #             }
        
# # # # # # #         return optimization_data
        
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error optimizing resume: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to optimize resume: {str(e)}")

# # # # # # # @app.post("/api/resume/skills/suggest")
# # # # # # # async def suggest_skills(input_data: SkillSuggestionInput):
# # # # # # #     """Suggest relevant skills for a profession"""
# # # # # # #     try:
# # # # # # #         profession = input_data.profession
# # # # # # #         current_skills = input_data.current_skills
        
# # # # # # #         prompt = f"""
# # # # # # #         Suggest 8-10 relevant skills for someone with the profession: "{profession}"
        
# # # # # # #         Current skills they already have: {', '.join(current_skills)}
        
# # # # # # #         Provide skills that would complement their existing skillset and are in-demand for this profession.
# # # # # # #         Focus on both technical and soft skills that are relevant.
        
# # # # # # #         Return only a JSON array of skill names:
# # # # # # #         ["skill1", "skill2", "skill3", ...]
# # # # # # #         """
        
# # # # # # #         response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[{"role": "user", "content": prompt}],
# # # # # # #             max_tokens=500,
# # # # # # #             temperature=0.3
# # # # # # #         )
        
# # # # # # #         result = response.choices[0].message.content or "[]"
        
# # # # # # #         # Clean the response
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             suggested_skills = json.loads(result)
# # # # # # #             if not isinstance(suggested_skills, list):
# # # # # # #                 suggested_skills = []
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             suggested_skills = []
        
# # # # # # #         return {"suggested_skills": suggested_skills}
        
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error suggesting skills: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to suggest skills: {str(e)}")

# # # # # # # @app.post("/api/resume/summary")
# # # # # # # async def generate_resume_summary_v2(input_data: ResumeInput):
# # # # # # #     """Generate resume summary (alternative endpoint)"""
# # # # # # #     return await generate_resume_summary(input_data)

# # # # # # # # --- Chatbot Endpoint ---

# # # # # # # # Add this Pydantic model near your other models
# # # # # # # class ChatRequest(BaseModel):
# # # # # # #     message: str
# # # # # # #     session_id: Optional[str] = None
# # # # # # #     context: Optional[dict] = None  # For resume_data

# # # # # # # # In-memory chat history store (temporary, replace with database for production)
# # # # # # # chat_history: Dict[str, list] = {}

# # # # # # # @app.post("/api/chatbot")
# # # # # # # async def chatbot(request: ChatRequest):
    
# # # # # # #     try:
# # # # # # #         # Get session_id or use default
# # # # # # #         session_id = request.session_id or "default_session"
        
# # # # # # #         # Initialize chat history for session if not exists
# # # # # # #         if session_id not in chat_history:
# # # # # # #             chat_history[session_id] = []

# # # # # # #         # Get resume data from context (if provided)
# # # # # # #         resume_data = request.context.get("resume_data", {}) if request.context else {}
# # # # # # #         resume_summary = (
# # # # # # #             f"Name: {resume_data.get('name', '')}, "
# # # # # # #             f"Skills: {', '.join(resume_data.get('skills', []))}, "
# # # # # # #             f"Education: {', '.join(resume_data.get('education', []))}"
# # # # # # #         ) if resume_data else "No resume data provided."

# # # # # # #         # Prepare chat history for prompt
# # # # # # #         history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[session_id]])

# # # # # # #         # Construct prompt
# # # # # # #         prompt = f"""
# # # # # # # You are the HG Resume Builder assistant, created by Hadiqa Gohar. Answer the user's question based on the provided resume data, chat history, and the following context:

# # # # # # # Context: HG Resume Builder offers AI-powered resume creation with three featured templates: Chameleon Pro Resume (ATS-friendly, customizable colors), Modern Professional (two-column layout, timeline design), and Creative Sidebar (sidebar design, gradient colors). Users can enhance CVs, export PDFs, and get expert feedback.

# # # # # # # Resume Data: {resume_summary}
# # # # # # # Chat History: {history_str}
# # # # # # # Question: {request.message}
# # # # # # # Answer in a concise, professional, and ATS-friendly manner. Provide 2-3 relevant suggestions for follow-up questions.
# # # # # # # """

# # # # # # #         # Call Gemini model using the existing external_client
# # # # # # #         response = await external_client.chat.completions.create(
# # # # # # #             model="gemini-2.0-flash",
# # # # # # #             messages=[{"role": "user", "content": prompt}],
# # # # # # #             max_tokens=500,
# # # # # # #             temperature=0.3
# # # # # # #         )

# # # # # # #         answer = response.choices[0].message.content or "Sorry, I couldn't generate a response."

# # # # # # #         # Update chat history
# # # # # # #         chat_history[session_id].append({"role": "user", "content": request.message})
# # # # # # #         chat_history[session_id].append({"role": "assistant", "content": answer})

# # # # # # #         # Keep history manageable (e.g., last 10 messages)
# # # # # # #         if len(chat_history[session_id]) > 10:
# # # # # # #             chat_history[session_id] = chat_history[session_id][-10:]

# # # # # # #         # Return response compatible with frontend
# # # # # # #         return {
# # # # # # #             "response": answer,
# # # # # # #             "type": "answer",
# # # # # # #             "sources": [],  # Add sources if you integrate a knowledge base later
# # # # # # #             "suggestions": [
# # # # # # #                 "How can I improve my resume summary?",
# # # # # # #                 "What skills should I add?",
# # # # # # #                 "Show me templates"
# # # # # # #             ],
# # # # # # #             "timestamp": datetime.now().isoformat(),
# # # # # # #             "session_id": session_id
# # # # # # #         }

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error in chatbot endpoint: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to process chatbot query: {str(e)}")

# # # # # # # @app.post("/api/chatbot/session/clear")
# # # # # # # async def clear_session(request: ChatRequest):
# # # # # # #     try:
# # # # # # #         session_id = request.session_id or "default_session"
# # # # # # #         if session_id in chat_history:
# # # # # # #             del chat_history[session_id]
# # # # # # #         return {"message": "Session cleared successfully"}
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error clearing session: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")
    

# # # # # # # @app.get("/")
# # # # # # # async def root():
    
# # # # # # #     return {"message": "FastAPI resume backend with Gemini is running."}


# # # # # # # import os
# # # # # # # import json
# # # # # # # from datetime import datetime
# # # # # # # from fastapi import FastAPI, HTTPException, Request
# # # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # # from pydantic import BaseModel
# # # # # # # from dotenv import load_dotenv
# # # # # # # from agents import Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
# # # # # # # from typing import Optional, Dict

# # # # # # # # Load environment variables from .env file
# # # # # # # load_dotenv()

# # # # # # # app = FastAPI()

# # # # # # # # --- CORS Configuration ---
# # # # # # # app.add_middleware(
# # # # # # #     CORSMiddleware,
# # # # # # #     allow_origins=["*"],  # For development, allow all origins. Change in production.
# # # # # # #     allow_credentials=True,
# # # # # # #     allow_methods=["*"],
# # # # # # #     allow_headers=["*"],
# # # # # # # )

# # # # # # # # --- Environment Variable Loading ---
# # # # # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # # # # # # # --- AI Agent Initialization (Gemini via OpenAI SDK compatibility) ---
# # # # # # # external_client = AsyncOpenAI(
# # # # # # #     api_key=GEMINI_API_KEY,
# # # # # # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # # # # # )

# # # # # # # model = OpenAIChatCompletionsModel(
# # # # # # #     openai_client=external_client,
# # # # # # #     model="gemini-2.0-flash",
# # # # # # # )

# # # # # # # config = RunConfig(
# # # # # # #     model=model,
# # # # # # #     model_provider=external_client,
# # # # # # #     instructions="You are Medicura-AI Health assistant, created by Hadiqa Gohar. Provide concise, professional responses to health-related queries, including symptom analysis, drug interaction checks, medical term explanations, and general health advice. Avoid definitive diagnoses and always include a disclaimer to consult a healthcare professional.",
# # # # # # #     tracing_disabled=True,
# # # # # # # )

# # # # # # # # Initialize Runner
# # # # # # # runner = Runner()

# # # # # # # # Pydantic models for request body
# # # # # # # class SymptomInput(BaseModel):
# # # # # # #     symptoms: list[str]
# # # # # # #     duration: Optional[str] = None
# # # # # # #     severity: Optional[str] = None

# # # # # # # class DrugInteractionInput(BaseModel):
# # # # # # #     medications: list[str]

# # # # # # # class MedicalTermInput(BaseModel):
# # # # # # #     term: str

# # # # # # # class HealthQueryInput(BaseModel):
# # # # # # #     query: str
# # # # # # #     context: Optional[dict] = None

# # # # # # # class ChatRequest(BaseModel):
# # # # # # #     message: str
# # # # # # #     session_id: Optional[str] = None
# # # # # # #     context: Optional[dict] = None  # For health-related context

# # # # # # # # In-memory chat history store (temporary, replace with database for production)
# # # # # # # chat_history: Dict[str, list] = {}

# # # # # # # @app.post("/api/health/symptoms")
# # # # # # # async def analyze_symptoms(input_data: SymptomInput):
# # # # # # #     """Analyze symptoms provided by the user"""
# # # # # # #     try:
# # # # # # #         symptoms_str = ", ".join(input_data.symptoms)
# # # # # # #         duration = input_data.duration or "not specified"
# # # # # # #         severity = input_data.severity or "not specified"

# # # # # # #         userquestion = f"""
# # # # # # #         Analyze the following symptoms and provide a concise, professional response with possible conditions, general advice, and a disclaimer to consult a healthcare professional. Do not provide a definitive diagnosis.

# # # # # # #         Symptoms: {symptoms_str}
# # # # # # #         Duration: {duration}
# # # # # # #         Severity: {severity}

# # # # # # #         Return a JSON response with:
# # # # # # #         - possible_conditions: List of potential conditions (3-5)
# # # # # # #         - general_advice: List of general health recommendations (3-5)
# # # # # # #         - disclaimer: A disclaimer about consulting a professional
# # # # # # #         """
        
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion=userquestion,
# # # # # # #             config=config
# # # # # # #         )

# # # # # # #         result = response.get('content', '{}')
        
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             analysis_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             analysis_data = {
# # # # # # #                 "possible_conditions": [],
# # # # # # #                 "general_advice": ["Consult a healthcare professional for accurate diagnosis."],
# # # # # # #                 "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
# # # # # # #             }

# # # # # # #         return analysis_data

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error analyzing symptoms: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to analyze symptoms: {str(e)}")

# # # # # # # @app.post("/api/health/drug-interactions")
# # # # # # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # # # # # #     """Check potential drug interactions"""
# # # # # # #     try:
# # # # # # #         medications_str = ", ".join(input_data.medications)

# # # # # # #         userquestion = f"""
# # # # # # #         Check for potential interactions between the following medications: {medications_str}. Provide a concise response with potential interactions, precautions, and a disclaimer to consult a healthcare professional.

# # # # # # #         Return a JSON response with:
# # # # # # #         - interactions: List of potential interactions (if any)
# # # # # # #         - precautions: List of precautions to take
# # # # # # #         - disclaimer: A disclaimer about consulting a professional
# # # # # # #         """
        
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion=userquestion,
# # # # # # #             config=config
# # # # # # #         )

# # # # # # #         result = response.get('content', '{}')
        
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             interaction_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             interaction_data = {
# # # # # # #                 "interactions": [],
# # # # # # #                 "precautions": ["Consult a healthcare professional before combining medications."],
# # # # # # #                 "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
# # # # # # #             }

# # # # # # #         return interaction_data

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error checking drug interactions: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to check drug interactions: {str(e)}")

# # # # # # # @app.post("/api/health/medical-term")
# # # # # # # async def explain_medical_term(input_data: MedicalTermInput):
# # # # # # #     """Explain a medical term"""
# # # # # # #     try:
# # # # # # #         term = input_data.term

# # # # # # #         userquestion = f"""
# # # # # # #         Provide a clear, concise explanation of the medical term "{term}" in layman's terms. Include its definition, common uses, and any relevant context.

# # # # # # #         Return a JSON response with:
# # # # # # #         - term: The medical term
# # # # # # #         - definition: A clear explanation
# # # # # # #         - common_uses: List of common contexts or uses
# # # # # # #         """
        
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion=userquestion,
# # # # # # #             config=config
# # # # # # #         )

# # # # # # #         result = response.get('content', '{}')
        
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             term_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             term_data = {
# # # # # # #                 "term": term,
# # # # # # #                 "definition": "Unable to provide definition at this time.",
# # # # # # #                 "common_uses": []
# # # # # # #             }

# # # # # # #         return term_data

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error explaining medical term: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to explain medical term: {str(e)}")

# # # # # # # @app.post("/api/health/query")
# # # # # # # async def general_health_query(input_data: HealthQueryInput):
# # # # # # #     """Handle general health-related queries"""
# # # # # # #     try:
# # # # # # #         query = input_data.query
# # # # # # #         context = input_data.context or {}

# # # # # # #         userquestion = f"""
# # # # # # #         Answer the following health-related query in a concise, professional manner. Provide general information and a disclaimer to consult a healthcare professional. Context: {json.dumps(context)}

# # # # # # #         Query: {query}

# # # # # # #         Return a JSON response with:
# # # # # # #         - answer: The response to the query
# # # # # # #         - disclaimer: A disclaimer about consulting a professional
# # # # # # #         """
        
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion=userquestion,
# # # # # # #             config=config
# # # # # # #         )

# # # # # # #         result = response.get('content', '{}')
        
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             query_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             query_data = {
# # # # # # #                 "answer": "Unable to process query at this time.",
# # # # # # #                 "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
# # # # # # #             }

# # # # # # #         return query_data

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error processing health query: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to process health query: {str(e)}")

# # # # # # # @app.get("/health")
# # # # # # # async def health_check():
# # # # # # #     """Health check endpoint"""
# # # # # # #     try:
# # # # # # #         # Test Gemini connection
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion="Hello",
# # # # # # #             config=config
# # # # # # #         )
# # # # # # #         gemini_status = "connected"
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Gemini connection failed: {e}")
# # # # # # #         gemini_status = "disconnected"
    
# # # # # # #     return {
# # # # # # #         "api_status": "healthy",
# # # # # # #         "gemini_status": gemini_status,
# # # # # # #         "timestamp": datetime.now().isoformat()
# # # # # # #     }

# # # # # # # @app.post("/api/chatbot")
# # # # # # # async def chatbot(request: ChatRequest):
# # # # # # #     """Handle chatbot interactions for health-related queries"""
# # # # # # #     try:
# # # # # # #         # Get session_id or use default
# # # # # # #         session_id = request.session_id or "default_session"
        
# # # # # # #         # Initialize chat history for session if not exists
# # # # # # #         if session_id not in chat_history:
# # # # # # #             chat_history[session_id] = []

# # # # # # #         # Get context (if provided)
# # # # # # #         context = request.context or {}
# # # # # # #         context_str = json.dumps(context) if context else "No additional context provided."

# # # # # # #         # Prepare chat history for prompt
# # # # # # #         history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[session_id]])

# # # # # # #         # Construct userquestion
# # # # # # #         userquestion = f"""
# # # # # # #         Answer the user's health-related question in a concise, professional manner. Provide general information, avoid definitive diagnoses, and include a disclaimer to consult a healthcare professional. Use the provided context and chat history to inform your response.

# # # # # # #         Context: {context_str}
# # # # # # #         Chat History: {history_str}
# # # # # # #         Question: {request.message}

# # # # # # #         Return a JSON response with:
# # # # # # #         - response: The answer to the question
# # # # # # #         - suggestions: 2-3 relevant follow-up question suggestions
# # # # # # #         - disclaimer: A disclaimer about consulting a professional
# # # # # # #         """

# # # # # # #         # Call Gemini model using Runner.run
# # # # # # #         response = await runner.run(
# # # # # # #             userquestion=userquestion,
# # # # # # #             config=config
# # # # # # #         )

# # # # # # #         result = response.get('content', '{}')
        
# # # # # # #         if result.startswith("```json"):
# # # # # # #             result = result[7:-3]
# # # # # # #         elif result.startswith("```"):
# # # # # # #             result = result[3:-3]
        
# # # # # # #         try:
# # # # # # #             response_data = json.loads(result)
# # # # # # #         except json.JSONDecodeError:
# # # # # # #             response_data = {
# # # # # # #                 "response": "Sorry, I couldn't generate a response.",
# # # # # # #                 "suggestions": [
# # # # # # #                     "Can you describe your symptoms?",
# # # # # # #                     "Do you need help with a medical term?",
# # # # # # #                     "Would you like to check drug interactions?"
# # # # # # #                 ],
# # # # # # #                 "disclaimer": "This information is for educational purposes only and not a substitute for professional medical advice."
# # # # # # #             }

# # # # # # #         # Update chat history
# # # # # # #         chat_history[session_id].append({"role": "user", "content": request.message})
# # # # # # #         chat_history[session_id].append({"role": "assistant", "content": response_data["response"]})

# # # # # # #         # Keep history manageable (e.g., last 10 messages)
# # # # # # #         if len(chat_history[session_id]) > 10:
# # # # # # #             chat_history[session_id] = chat_history[session_id][-10:]

# # # # # # #         # Return response compatible with frontend
# # # # # # #         return {
# # # # # # #             "response": response_data["response"],
# # # # # # #             "type": "answer",
# # # # # # #             "sources": [],  # Add sources if you integrate a knowledge base later
# # # # # # #             "suggestions": response_data["suggestions"],
# # # # # # #             "disclaimer": response_data["disclaimer"],
# # # # # # #             "timestamp": datetime.now().isoformat(),
# # # # # # #             "session_id": session_id
# # # # # # #         }

# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error in chatbot endpoint: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to process chatbot query: {str(e)}")

# # # # # # # @app.post("/api/chatbot/session/clear")
# # # # # # # async def clear_session(request: ChatRequest):
# # # # # # #     """Clear chatbot session history"""
# # # # # # #     try:
# # # # # # #         session_id = request.session_id or "default_session"
# # # # # # #         if session_id in chat_history:
# # # # # # #             del chat_history[session_id]
# # # # # # #         return {"message": "Session cleared successfully"}
# # # # # # #     except Exception as e:
# # # # # # #         print(f"Error clearing session: {str(e)}")
# # # # # # #         raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

# # # # # # # @app.get("/")
# # # # # # # async def root():
# # # # # # #     """Root endpoint"""
# # # # # # #     return {"message": "FastAPI health backend with Gemini is running."}


# # # # # # # -------------------------------------------------------------------------------------------------------------------------------------

# # # # # # import os
# # # # # # import json
# # # # # # import asyncio
# # # # # # from datetime import datetime
# # # # # # from fastapi import FastAPI, HTTPException
# # # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # # from pydantic import BaseModel
# # # # # # from dotenv import load_dotenv
# # # # # # from typing import Optional, Dict, List
# # # # # # import logging
# # # # # # from contextlib import asynccontextmanager
# # # # # # import aiohttp

# # # # # # # Configure logging
# # # # # # logging.basicConfig(level=logging.INFO)
# # # # # # logger = logging.getLogger(__name__)

# # # # # # # Load environment variables from .env file
# # # # # # load_dotenv()

# # # # # # # Global variable for chat history
# # # # # # chat_history: Dict[str, List[dict]] = {}

# # # # # # @asynccontextmanager
# # # # # # async def lifespan(app: FastAPI):
# # # # # #     # Startup
# # # # # #     logger.info("Starting Medicura-AI Health Assistant")
# # # # # #     yield
# # # # # #     # Shutdown
# # # # # #     logger.info("Shutting down Medicura-AI Health Assistant")

# # # # # # app = FastAPI(title="Medicura-AI Health Assistant", 
# # # # # #               description="AI-powered health assistant for symptom analysis and medical queries",
# # # # # #               version="1.0.0",
# # # # # #               lifespan=lifespan)

# # # # # # # --- CORS Configuration ---
# # # # # # app.add_middleware(
# # # # # #     CORSMiddleware,
# # # # # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
# # # # # #     allow_credentials=True,
# # # # # #     allow_methods=["*"],
# # # # # #     allow_headers=["*"],
# # # # # # )

# # # # # # # --- Environment Variable Loading ---
# # # # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # # # if not GEMINI_API_KEY:
# # # # # #     logger.error("GEMINI_API_KEY not found in environment variables")

# # # # # # # Direct Gemini API integration (more reliable)
# # # # # # async def call_gemini_direct(prompt: str, max_retries: int = 3) -> str:
# # # # # #     """Call Gemini API directly using REST"""
# # # # # #     if not GEMINI_API_KEY:
# # # # # #         return await generate_fallback_response(prompt)
    
# # # # # #     url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
# # # # # #     headers = {
# # # # # #         "Content-Type": "application/json",
# # # # # #         "x-goog-api-key": GEMINI_API_KEY
# # # # # #     }
    
# # # # # #     payload = {
# # # # # #         "contents": [{
# # # # # #             "parts": [{
# # # # # #                 "text": f"""You are Medicura-AI Health Assistant, created by Hadiqa Gohar. 
# # # # # #                 Provide accurate, professional health information with proper disclaimers.
                
# # # # # #                 {prompt}
                
# # # # # #                 Always return valid JSON with this structure:
# # # # # #                 {{
# # # # # #                     "response": "detailed answer here",
# # # # # #                     "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
# # # # # #                     "disclaimer": "consult healthcare professional disclaimer"
# # # # # #                 }}"""
# # # # # #             }]
# # # # # #         }],
# # # # # #         "generationConfig": {
# # # # # #             "temperature": 0.7,
# # # # # #             "maxOutputTokens": 1024,
# # # # # #         }
# # # # # #     }
    
# # # # # #     for attempt in range(max_retries):
# # # # # #         try:
# # # # # #             async with aiohttp.ClientSession() as session:
# # # # # #                 async with session.post(url, headers=headers, json=payload) as response:
# # # # # #                     if response.status == 200:
# # # # # #                         data = await response.json()
# # # # # #                         if 'candidates' in data and data['candidates']:
# # # # # #                             text = data['candidates'][0]['content']['parts'][0]['text']
# # # # # #                             return text
# # # # # #                     else:
# # # # # #                         logger.error(f"Gemini API error: {response.status}")
# # # # # #         except Exception as e:
# # # # # #             if attempt == max_retries - 1:
# # # # # #                 logger.error(f"Failed to call Gemini API: {e}")
# # # # # #                 return await generate_fallback_response(prompt)
# # # # # #             await asyncio.sleep(1 * (attempt + 1))
    
# # # # # #     return await generate_fallback_response(prompt)

# # # # # # async def generate_fallback_response(prompt: str) -> str:
# # # # # #     """Generate intelligent fallback responses based on query content"""
# # # # # #     prompt_lower = prompt.lower()
    
# # # # # #     # Medical term explanations
# # # # # #     if any(term in prompt_lower for term in ['glutathione', 'glutathion', 'antioxidant']):
# # # # # #         return json.dumps({
# # # # # #             "response": "Glutathione is a powerful antioxidant produced naturally in the body. It helps protect cells from damage and supports detoxification. Common uses include supporting liver health, immune function, and antioxidant defense. Potential side effects may include abdominal discomfort, allergic reactions in sensitive individuals, or interactions with certain medications. Always consult a healthcare provider before taking glutathione supplements.",
# # # # # #             "suggestions": [
# # # # # #                 "What are the benefits of glutathione?",
# # # # # #                 "How to naturally increase glutathione levels?",
# # # # # #                 "Glutathione dosage recommendations"
# # # # # #             ],
# # # # # #             "disclaimer": "This information is for educational purposes. Consult a healthcare professional before taking any supplements."
# # # # # #         })
    
# # # # # #     # Fever treatment
# # # # # #     elif any(term in prompt_lower for term in ['fever', 'temperature', 'pyrexia']):
# # # # # #         return json.dumps({
# # # # # #             "response": "For fever management, common approaches include: 1) Rest and hydration with water, electrolytes 2) Over-the-counter medications like acetaminophen (Tylenol) or ibuprofen (Advil) as directed 3) Cool compresses and light clothing 4) Monitoring temperature regularly. Seek medical attention if fever is high (over 103°F/39.4°C), persists more than 3 days, or is accompanied by severe symptoms.",
# # # # # #             "suggestions": [
# # # # # #                 "When to seek medical help for fever?",
# # # # # #                 "Natural remedies for fever reduction",
# # # # # #                 "Fever in children - special considerations"
# # # # # #             ],
# # # # # #             "disclaimer": "This is general information. Always follow healthcare provider advice for fever management."
# # # # # #         })
    
# # # # # #     # Medication questions
# # # # # #     elif any(term in prompt_lower for term in ['medicine', 'medication', 'drug', 'pill', 'tablet']):
# # # # # #         return json.dumps({
# # # # # #             "response": "I can provide general information about medications, but specific medical advice should come from healthcare professionals. Please consult your doctor or pharmacist for personalized medication guidance, dosage information, and potential interactions.",
# # # # # #             "suggestions": [
# # # # # #                 "How to take medications safely?",
# # # # # #                 "Common drug interactions to avoid",
# # # # # #                 "Storage and handling of medications"
# # # # # #             ],
# # # # # #             "disclaimer": "Always follow healthcare professional advice regarding medications."
# # # # # #         })
    
# # # # # #     # General health questions
# # # # # #     else:
# # # # # #         return json.dumps({
# # # # # #             "response": "Hello! I'm Medicura-AI Health Assistant, created by Hadiqa Gohar. I specialize in health-related questions including symptom analysis, medication information, medical term explanations, and general wellness advice. How can I assist you with your health concerns today?",
# # # # # #             "suggestions": [
# # # # # #                 "Tell me about your symptoms",
# # # # # #                 "Ask about medication side effects",
# # # # # #                 "Explain a medical condition",
# # # # # #                 "General health and wellness tips"
# # # # # #             ],
# # # # # #             "disclaimer": "I provide health information for educational purposes. Always consult healthcare professionals for medical advice."
# # # # # #         })

# # # # # # def extract_json_from_response(response_text: str) -> dict:
# # # # # #     """Extract JSON from response text"""
# # # # # #     if not response_text:
# # # # # #         return {}
    
# # # # # #     # Clean the response
# # # # # #     cleaned = response_text.strip()
    
# # # # # #     # Remove JSON code blocks if present
# # # # # #     if cleaned.startswith("```json"):
# # # # # #         cleaned = cleaned[7:].rsplit("```", 1)[0].strip()
# # # # # #     elif cleaned.startswith("```"):
# # # # # #         cleaned = cleaned[3:].rsplit("```", 1)[0].strip()
    
# # # # # #     try:
# # # # # #         return json.loads(cleaned)
# # # # # #     except json.JSONDecodeError:
# # # # # #         # Try to find JSON object in the text
# # # # # #         try:
# # # # # #             start = cleaned.find('{')
# # # # # #             end = cleaned.rfind('}') + 1
# # # # # #             if start != -1 and end != -1:
# # # # # #                 json_str = cleaned[start:end]
# # # # # #                 return json.loads(json_str)
# # # # # #         except:
# # # # # #             pass
        
# # # # # #         logger.warning(f"Failed to parse JSON response: {cleaned[:100]}...")
# # # # # #         return {}

# # # # # # # Pydantic models for request body
# # # # # # class SymptomInput(BaseModel):
# # # # # #     symptoms: List[str]
# # # # # #     duration: Optional[str] = None
# # # # # #     severity: Optional[str] = None

# # # # # # class DrugInteractionInput(BaseModel):
# # # # # #     medications: List[str]

# # # # # # class MedicalTermInput(BaseModel):
# # # # # #     term: str

# # # # # # class HealthQueryInput(BaseModel):
# # # # # #     query: str
# # # # # #     context: Optional[dict] = None

# # # # # # class ChatRequest(BaseModel):
# # # # # #     message: str
# # # # # #     session_id: Optional[str] = None
# # # # # #     context: Optional[dict] = None

# # # # # # class ClearSessionRequest(BaseModel):
# # # # # #     session_id: Optional[str] = None

# # # # # # # API endpoints
# # # # # # @app.post("/api/health/symptoms")
# # # # # # async def analyze_symptoms(input_data: SymptomInput):
# # # # # #     """Analyze symptoms provided by the user"""
# # # # # #     try:
# # # # # #         symptoms_str = ", ".join(input_data.symptoms)
# # # # # #         duration = input_data.duration or "not specified"
# # # # # #         severity = input_data.severity or "not specified"

# # # # # #         prompt = f"""
# # # # # #         Analyze these symptoms for potential conditions and provide advice:
# # # # # #         Symptoms: {symptoms_str}
# # # # # #         Duration: {duration}
# # # # # #         Severity: {severity}

# # # # # #         Return JSON with:
# # # # # #         - possible_conditions: list of 3-5 potential conditions
# # # # # #         - general_advice: list of 3-5 recommendations
# # # # # #         - disclaimer: consultation disclaimer
# # # # # #         """

# # # # # #         result = await call_gemini_direct(prompt)
# # # # # #         analysis_data = extract_json_from_response(result)

# # # # # #         if not analysis_data:
# # # # # #             analysis_data = {
# # # # # #                 "possible_conditions": ["Consult healthcare professional for proper diagnosis"],
# # # # # #                 "general_advice": ["Rest adequately", "Stay hydrated", "Monitor symptoms", "Avoid self-medication"],
# # # # # #                 "disclaimer": "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# # # # # #             }

# # # # # #         return analysis_data

# # # # # #     except Exception as e:
# # # # # #         logger.error(f"Error analyzing symptoms: {str(e)}")
# # # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # # @app.post("/api/chatbot")
# # # # # # async def chatbot(request: ChatRequest):
# # # # # #     """Handle chatbot interactions"""
# # # # # #     try:
# # # # # #         session_id = request.session_id or "default_session"
        
# # # # # #         if session_id not in chat_history:
# # # # # #             chat_history[session_id] = []

# # # # # #         prompt = f"""
# # # # # #         User question: {request.message}
        
# # # # # #         Provide a helpful, professional health response. Be specific and accurate.
# # # # # #         Return valid JSON with response, suggestions, and disclaimer.
# # # # # #         """

# # # # # #         result = await call_gemini_direct(prompt)
# # # # # #         response_data = extract_json_from_response(result)

# # # # # #         if not response_data:
# # # # # #             # Generate intelligent fallback based on query content
# # # # # #             fallback_result = await generate_fallback_response(request.message)
# # # # # #             response_data = extract_json_from_response(fallback_result)

# # # # # #         # Update chat history
# # # # # #         chat_history[session_id].extend([
# # # # # #             {"role": "user", "content": request.message},
# # # # # #             {"role": "assistant", "content": response_data["response"]}
# # # # # #         ])
# # # # # #         # Keep last 10 messages
# # # # # #         chat_history[session_id] = chat_history[session_id][-10:]

# # # # # #         return {
# # # # # #             "response": response_data["response"],
# # # # # #             "suggestions": response_data.get("suggestions", []),
# # # # # #             "disclaimer": response_data.get("disclaimer", ""),
# # # # # #             "timestamp": datetime.now().isoformat(),
# # # # # #             "session_id": session_id
# # # # # #         }

# # # # # #     except Exception as e:
# # # # # #         logger.error(f"Error in chatbot: {str(e)}")
# # # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # # @app.post("/api/chatbot/session/clear")
# # # # # # async def clear_session(request: ClearSessionRequest):
# # # # # #     """Clear chatbot session history"""
# # # # # #     try:
# # # # # #         session_id = request.session_id or "default_session"
# # # # # #         if session_id in chat_history:
# # # # # #             del chat_history[session_id]
# # # # # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # # # # #     except Exception as e:
# # # # # #         logger.error(f"Error clearing session: {str(e)}")
# # # # # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # # # # @app.get("/health")
# # # # # # async def health_check():
# # # # # #     """Health check endpoint"""
# # # # # #     return {
# # # # # #         "status": "healthy",
# # # # # #         "timestamp": datetime.now().isoformat(),
# # # # # #         "version": "1.0.0",
# # # # # #         "gemini_available": bool(GEMINI_API_KEY)
# # # # # #     }

# # # # # # @app.get("/")
# # # # # # async def root():
# # # # # #     """Root endpoint"""
# # # # # #     return {
# # # # # #         "message": "Medicura-AI Health Assistant API",
# # # # # #         "version": "1.0.0",
# # # # # #         "endpoints": [
# # # # # #             "/api/chatbot - Health chatbot",
# # # # # #             "/api/health/symptoms - Symptom analysis",
# # # # # #             "/docs - API documentation"
# # # # # #         ]
# # # # # #     }

# # # # # # if __name__ == "__main__":
# # # # # #     import uvicorn
# # # # # #     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# # # # # import os
# # # # # import json
# # # # # import asyncio
# # # # # from datetime import datetime
# # # # # from fastapi import FastAPI, HTTPException
# # # # # from fastapi.middleware.cors import CORSMiddleware
# # # # # from pydantic import BaseModel
# # # # # from dotenv import load_dotenv
# # # # # from typing import Optional, Dict, List
# # # # # import logging
# # # # # from contextlib import asynccontextmanager
# # # # # import aiohttp

# # # # # # Configure logging
# # # # # logging.basicConfig(level=logging.INFO)
# # # # # logger = logging.getLogger(__name__)

# # # # # # Load environment variables from .env file
# # # # # load_dotenv()

# # # # # # Global variable for chat history
# # # # # chat_history: Dict[str, List[dict]] = {}

# # # # # @asynccontextmanager
# # # # # async def lifespan(app: FastAPI):
# # # # #     # Startup
# # # # #     logger.info("Starting Medicura-AI Health Assistant")
# # # # #     yield
# # # # #     # Shutdown
# # # # #     logger.info("Shutting down Medicura-AI Health Assistant")

# # # # # app = FastAPI(title="Medicura-AI Health Assistant", 
# # # # #               description="AI-powered health assistant for symptom analysis and medical queries",
# # # # #               version="1.0.0",
# # # # #               lifespan=lifespan)

# # # # # # --- CORS Configuration ---
# # # # # app.add_middleware(
# # # # #     CORSMiddleware,
# # # # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
# # # # #     allow_credentials=True,
# # # # #     allow_methods=["*"],
# # # # #     allow_headers=["*"],
# # # # # )

# # # # # # --- Environment Variable Loading ---
# # # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # # if not GEMINI_API_KEY:
# # # # #     logger.error("GEMINI_API_KEY not found in environment variables")

# # # # # # Direct Gemini API integration (more reliable)
# # # # # async def call_gemini_direct(prompt: str, max_retries: int = 3) -> str:
# # # # #     """Call Gemini API directly using REST"""
# # # # #     if not GEMINI_API_KEY:
# # # # #         return await generate_fallback_response(prompt)
    
# # # # #     url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
# # # # #     headers = {
# # # # #         "Content-Type": "application/json",
# # # # #         "x-goog-api-key": GEMINI_API_KEY
# # # # #     }
    
# # # # #     payload = {
# # # # #         "contents": [{
# # # # #             "parts": [{
# # # # #                 "text": f"""You are Medicura-AI Health Assistant, created by Hadiqa Gohar. 
# # # # #                 Provide accurate, professional health information with proper disclaimers.
                
# # # # #                 {prompt}
                
# # # # #                 Always return valid JSON with this structure:
# # # # #                 {{
# # # # #                     "response": "detailed answer here",
# # # # #                     "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
# # # # #                     "disclaimer": "consult healthcare professional disclaimer"
# # # # #                 }}"""
# # # # #             }]
# # # # #         }],
# # # # #         "generationConfig": {
# # # # #             "temperature": 0.7,
# # # # #             "maxOutputTokens": 1024,
# # # # #         }
# # # # #     }
    
# # # # #     for attempt in range(max_retries):
# # # # #         try:
# # # # #             async with aiohttp.ClientSession() as session:
# # # # #                 async with session.post(url, headers=headers, json=payload) as response:
# # # # #                     if response.status == 200:
# # # # #                         data = await response.json()
# # # # #                         if 'candidates' in data and data['candidates']:
# # # # #                             text = data['candidates'][0]['content']['parts'][0]['text']
# # # # #                             return text
# # # # #                     else:
# # # # #                         logger.error(f"Gemini API error: {response.status}")
# # # # #         except Exception as e:
# # # # #             if attempt == max_retries - 1:
# # # # #                 logger.error(f"Failed to call Gemini API: {e}")
# # # # #                 return await generate_fallback_response(prompt)
# # # # #             await asyncio.sleep(1 * (attempt + 1))
    
# # # # #     return await generate_fallback_response(prompt)

# # # # # async def generate_fallback_response(prompt: str) -> str:
# # # # #     """Generate intelligent fallback responses based on query content"""
# # # # #     prompt_lower = prompt.lower()
    
# # # # #     # Medical term explanations
# # # # #     if any(term in prompt_lower for term in ['glutathione', 'glutathion', 'antioxidant']):
# # # # #         return json.dumps({
# # # # #             "response": "Glutathione is a powerful antioxidant produced naturally in the body. It helps protect cells from damage and supports detoxification. Common uses include supporting liver health, immune function, and antioxidant defense. Potential side effects may include abdominal discomfort, allergic reactions in sensitive individuals, or interactions with certain medications. Always consult a healthcare provider before taking glutathione supplements.",
# # # # #             "suggestions": [
# # # # #                 "What are the benefits of glutathione?",
# # # # #                 "How to naturally increase glutathione levels?",
# # # # #                 "Glutathione dosage recommendations"
# # # # #             ],
# # # # #             "disclaimer": "This information is for educational purposes. Consult a healthcare professional before taking any supplements."
# # # # #         })
    
# # # # #     # Fever treatment
# # # # #     elif any(term in prompt_lower for term in ['fever', 'temperature', 'pyrexia']):
# # # # #         return json.dumps({
# # # # #             "response": "For fever management, common approaches include: 1) Rest and hydration with water, electrolytes 2) Over-the-counter medications like acetaminophen (Tylenol) or ibuprofen (Advil) as directed 3) Cool compresses and light clothing 4) Monitoring temperature regularly. Seek medical attention if fever is high (over 103°F/39.4°C), persists more than 3 days, or is accompanied by severe symptoms.",
# # # # #             "suggestions": [
# # # # #                 "When to seek medical help for fever?",
# # # # #                 "Natural remedies for fever reduction",
# # # # #                 "Fever in children - special considerations"
# # # # #             ],
# # # # #             "disclaimer": "This is general information. Always follow healthcare provider advice for fever management."
# # # # #         })
    
# # # # #     # Medication questions
# # # # #     elif any(term in prompt_lower for term in ['medicine', 'medication', 'drug', 'pill', 'tablet']):
# # # # #         return json.dumps({
# # # # #             "response": "I can provide general information about medications, but specific medical advice should come from healthcare professionals. Please consult your doctor or pharmacist for personalized medication guidance, dosage information, and potential interactions.",
# # # # #             "suggestions": [
# # # # #                 "How to take medications safely?",
# # # # #                 "Common drug interactions to avoid",
# # # # #                 "Storage and handling of medications"
# # # # #             ],
# # # # #             "disclaimer": "Always follow healthcare professional advice regarding medications."
# # # # #         })
    
# # # # #     # General health questions
# # # # #     else:
# # # # #         return json.dumps({
# # # # #             "response": "Hello! I'm Medicura-AI Health Assistant, created by Hadiqa Gohar. I specialize in health-related questions including symptom analysis, medication information, medical term explanations, and general wellness advice. How can I assist you with your health concerns today?",
# # # # #             "suggestions": [
# # # # #                 "Tell me about your symptoms",
# # # # #                 "Ask about medication side effects",
# # # # #                 "Explain a medical condition",
# # # # #                 "General health and wellness tips"
# # # # #             ],
# # # # #             "disclaimer": "I provide health information for educational purposes. Always consult healthcare professionals for medical advice."
# # # # #         })

# # # # # def extract_json_from_response(response_text: str) -> dict:
# # # # #     """Extract JSON from response text"""
# # # # #     if not response_text:
# # # # #         return {}
    
# # # # #     # Clean the response
# # # # #     cleaned = response_text.strip()
    
# # # # #     # Remove JSON code blocks if present
# # # # #     if cleaned.startswith("```json"):
# # # # #         cleaned = cleaned[7:].rsplit("```", 1)[0].strip()
# # # # #     elif cleaned.startswith("```"):
# # # # #         cleaned = cleaned[3:].rsplit("```", 1)[0].strip()
    
# # # # #     try:
# # # # #         return json.loads(cleaned)
# # # # #     except json.JSONDecodeError:
# # # # #         # Try to find JSON object in the text
# # # # #         try:
# # # # #             start = cleaned.find('{')
# # # # #             end = cleaned.rfind('}') + 1
# # # # #             if start != -1 and end != -1:
# # # # #                 json_str = cleaned[start:end]
# # # # #                 return json.loads(json_str)
# # # # #         except:
# # # # #             pass
        
# # # # #         logger.warning(f"Failed to parse JSON response: {cleaned[:100]}...")
# # # # #         return {}

# # # # # # Pydantic models for request body
# # # # # class SymptomInput(BaseModel):
# # # # #     symptoms: List[str]
# # # # #     duration: Optional[str] = None
# # # # #     severity: Optional[str] = None
# # # # #     age: Optional[int] = None
# # # # #     gender: Optional[str] = None
# # # # #     existing_conditions: Optional[List[str]] = None
# # # # #     medications: Optional[List[str]] = None

# # # # # class DrugInteractionInput(BaseModel):
# # # # #     medications: List[str]

# # # # # class MedicalTermInput(BaseModel):
# # # # #     term: str

# # # # # class HealthQueryInput(BaseModel):
# # # # #     query: str
# # # # #     context: Optional[dict] = None

# # # # # class ChatRequest(BaseModel):
# # # # #     message: str
# # # # #     session_id: Optional[str] = None
# # # # #     context: Optional[dict] = None

# # # # # class ClearSessionRequest(BaseModel):
# # # # #     session_id: Optional[str] = None


# # # # # class DrugInteractionInput(BaseModel):
# # # # #     medications: List[str]
# # # # #     age: Optional[int] = None
# # # # #     gender: Optional[str] = None
# # # # #     existing_conditions: Optional[List[str]] = None
# # # # #     other_medications: Optional[List[str]] = None

# # # # # # API endpoints
# # # # # @app.post("/api/health/symptoms")
# # # # # async def analyze_symptoms(input_data: SymptomInput):
# # # # #     """Analyze symptoms provided by the user"""
# # # # #     try:
# # # # #         symptoms_str = ", ".join(input_data.symptoms)
# # # # #         duration = input_data.duration or "not specified"
# # # # #         severity = input_data.severity or "not specified"
# # # # #         age = input_data.age or "not specified"
# # # # #         gender = input_data.gender or "not specified"
# # # # #         existing_conditions = ", ".join(input_data.existing_conditions) if input_data.existing_conditions else "none"
# # # # #         medications = ", ".join(input_data.medications) if input_data.medications else "none"

# # # # #         prompt = f"""
# # # # #         Analyze these symptoms for potential conditions and provide advice:
# # # # #         Symptoms: {symptoms_str}
# # # # #         Duration: {duration}
# # # # #         Severity: {severity}
# # # # #         Age: {age}
# # # # #         Gender: {gender}
# # # # #         Existing medical conditions: {existing_conditions}
# # # # #         Current medications: {medications}

# # # # #         Return JSON with:
# # # # #         - possible_conditions: list of 3-5 potential conditions with brief explanations
# # # # #         - recommended_actions: list of 3-5 recommended actions
# # # # #         - when_to_seek_help: specific guidance on when to seek medical attention
# # # # #         - home_remedies: safe home remedies if applicable
# # # # #         - disclaimer: consultation disclaimer

# # # # #         Prioritize common conditions first. Be cautious and emphasize seeking professional help when appropriate.
# # # # #         """

# # # # #         result = await call_gemini_direct(prompt)
# # # # #         analysis_data = extract_json_from_response(result)

# # # # #         if not analysis_data:
# # # # #             analysis_data = {
# # # # #                 "possible_conditions": ["Consult healthcare professional for proper diagnosis"],
# # # # #                 "recommended_actions": ["Rest adequately", "Stay hydrated", "Monitor symptoms", "Avoid self-medication"],
# # # # #                 "when_to_seek_help": ["If symptoms worsen", "If fever persists beyond 3 days", "If difficulty breathing occurs"],
# # # # #                 "home_remedies": ["Warm salt water gargle for throat symptoms", "Adequate rest", "Proper hydration"],
# # # # #                 "disclaimer": "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# # # # #             }

# # # # #         return analysis_data

# # # # #     except Exception as e:
# # # # #         logger.error(f"Error analyzing symptoms: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/health/symptoms/quick")
# # # # # async def quick_symptom_check(symptoms: List[str]):
# # # # #     """Quick symptom check without detailed information"""
# # # # #     try:
# # # # #         if not symptoms:
# # # # #             raise HTTPException(status_code=400, detail="At least one symptom is required")
        
# # # # #         symptoms_str = ", ".join(symptoms)
        
# # # # #         prompt = f"""
# # # # #         Provide a quick assessment of these symptoms: {symptoms_str}
        
# # # # #         Return JSON with:
# # # # #         - urgency_level: low, medium, or high
# # # # #         - immediate_advice: 2-3 immediate steps to take
# # # # #         - should_seek_help: boolean indicating if medical attention is recommended
# # # # #         - disclaimer: consultation disclaimer
# # # # #         """
        
# # # # #         result = await call_gemini_direct(prompt)
# # # # #         analysis_data = extract_json_from_response(result)
        
# # # # #         if not analysis_data:
# # # # #             analysis_data = {
# # # # #                 "urgency_level": "medium",
# # # # #                 "immediate_advice": ["Rest and monitor symptoms", "Stay hydrated", "Avoid self-medication"],
# # # # #                 "should_seek_help": False,
# # # # #                 "disclaimer": "This is a preliminary assessment. Consult a healthcare professional for proper diagnosis."
# # # # #             }
            
# # # # #         return analysis_data
        
# # # # #     except Exception as e:
# # # # #         logger.error(f"Error in quick symptom check: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/chatbot")
# # # # # async def chatbot(request: ChatRequest):
# # # # #     """Handle chatbot interactions"""
# # # # #     try:
# # # # #         session_id = request.session_id or "default_session"
        
# # # # #         if session_id not in chat_history:
# # # # #             chat_history[session_id] = []

# # # # #         prompt = f"""
# # # # #         User question: {request.message}
        
# # # # #         Provide a helpful, professional health response. Be specific and accurate.
# # # # #         Return valid JSON with response, suggestions, and disclaimer.
# # # # #         """

# # # # #         result = await call_gemini_direct(prompt)
# # # # #         response_data = extract_json_from_response(result)

# # # # #         if not response_data:
# # # # #             # Generate intelligent fallback based on query content
# # # # #             fallback_result = await generate_fallback_response(request.message)
# # # # #             response_data = extract_json_from_response(fallback_result)

# # # # #         # Update chat history
# # # # #         chat_history[session_id].extend([
# # # # #             {"role": "user", "content": request.message},
# # # # #             {"role": "assistant", "content": response_data["response"]}
# # # # #         ])
# # # # #         # Keep last 10 messages
# # # # #         chat_history[session_id] = chat_history[session_id][-10:]

# # # # #         return {
# # # # #             "response": response_data["response"],
# # # # #             "suggestions": response_data.get("suggestions", []),
# # # # #             "disclaimer": response_data.get("disclaimer", ""),
# # # # #             "timestamp": datetime.now().isoformat(),
# # # # #             "session_id": session_id
# # # # #         }

# # # # #     except Exception as e:
# # # # #         logger.error(f"Error in chatbot: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/chatbot/session/clear")
# # # # # async def clear_session(request: ClearSessionRequest):
# # # # #     """Clear chatbot session history"""
# # # # #     try:
# # # # #         session_id = request.session_id or "default_session"
# # # # #         if session_id in chat_history:
# # # # #             del chat_history[session_id]
# # # # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # # # #     except Exception as e:
# # # # #         logger.error(f"Error clearing session: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # # # @app.get("/health")
# # # # # async def health_check():
# # # # #     """Health check endpoint"""
# # # # #     return {
# # # # #         "status": "healthy",
# # # # #         "timestamp": datetime.now().isoformat(),
# # # # #         "version": "1.0.0",
# # # # #         "gemini_available": bool(GEMINI_API_KEY)
# # # # #     }


# # # # # @app.post("/api/health/drug-interactions")
# # # # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # # # #     """Check for potential drug interactions"""
# # # # #     try:
# # # # #         if not input_data.medications or len(input_data.medications) == 0:
# # # # #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# # # # #         medications_str = ", ".join(input_data.medications)
# # # # #         age = input_data.age or "not specified"
# # # # #         gender = input_data.gender or "not specified"
# # # # #         existing_conditions = ", ".join(input_data.existing_conditions) if input_data.existing_conditions else "none"
# # # # #         other_medications = ", ".join(input_data.other_medications) if input_data.other_medications else "none"

# # # # #         prompt = f"""
# # # # #         Analyze potential drug interactions for these medications: {medications_str}
        
# # # # #         Patient information:
# # # # #         - Age: {age}
# # # # #         - Gender: {gender}
# # # # #         - Existing conditions: {existing_conditions}
# # # # #         - Other medications: {other_medications}
        
# # # # #         Return JSON with:
# # # # #         - interactions: list of potential interactions with severity (high, medium, low)
# # # # #         - recommendations: specific recommendations for each interaction
# # # # #         - alternative_options: suggested alternative medications if available
# # # # #         - general_advice: general medication safety advice
# # # # #         - disclaimer: consultation disclaimer
        
# # # # #         Be thorough and cautious. Highlight dangerous interactions clearly.
# # # # #         """

# # # # #         result = await call_gemini_direct(prompt)
# # # # #         interaction_data = extract_json_from_response(result)

# # # # #         if not interaction_data:
# # # # #             interaction_data = {
# # # # #                 "interactions": [
# # # # #                     {
# # # # #                         "medications": input_data.medications,
# # # # #                         "severity": "unknown",
# # # # #                         "description": "Unable to analyze interactions. Consult a healthcare professional."
# # # # #                     }
# # # # #                 ],
# # # # #                 "recommendations": ["Consult a pharmacist or doctor for detailed interaction analysis"],
# # # # #                 "alternative_options": [],
# # # # #                 "general_advice": [
# # # # #                     "Always inform your doctor about all medications you're taking",
# # # # #                     "Read medication labels carefully",
# # # # #                     "Don't start or stop medications without medical advice"
# # # # #                 ],
# # # # #                 "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medication advice."
# # # # #             }

# # # # #         return interaction_data

# # # # #     except Exception as e:
# # # # #         logger.error(f"Error checking drug interactions: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # # @app.post("/api/health/drug-info")
# # # # # async def get_drug_information(medication: str):
# # # # #     """Get information about a specific medication"""
# # # # #     try:
# # # # #         if not medication:
# # # # #             raise HTTPException(status_code=400, detail="Medication name is required")
        
# # # # #         prompt = f"""
# # # # #         Provide information about the medication: {medication}
        
# # # # #         Return JSON with:
# # # # #         - uses: primary uses and indications
# # # # #         - dosage: typical dosage information
# # # # #         - side_effects: common and serious side effects
# # # # #         - precautions: important precautions and warnings
# # # # #         - interactions: common drugs it interacts with
# # # # #         - disclaimer: consultation disclaimer
        
# # # # #         Be accurate and comprehensive. Include both brand and generic names if applicable.
# # # # #         """

# # # # #         result = await call_gemini_direct(prompt)
# # # # #         drug_data = extract_json_from_response(result)

# # # # #         if not drug_data:
# # # # #             drug_data = {
# # # # #                 "uses": ["Consult healthcare professional for accurate information"],
# # # # #                 "dosage": "Dosage varies based on condition and patient factors",
# # # # #                 "side_effects": ["Nausea", "Headache", "Dizziness", "Consult doctor for complete list"],
# # # # #                 "precautions": ["Take as prescribed", "Don't share medication", "Store properly"],
# # # # #                 "interactions": ["Many potential interactions - consult pharmacist"],
# # # # #                 "disclaimer": "This information is for educational purposes only. Always follow healthcare professional advice."
# # # # #             }

# # # # #         return drug_data

# # # # #     except Exception as e:
# # # # #         logger.error(f"Error getting drug information: {str(e)}")
# # # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")


# # # # # @app.get("/")
# # # # # async def root():
# # # # #     """Root endpoint"""
# # # # #     return {
# # # # #         "message": "Medicura-AI Health Assistant API",
# # # # #         "version": "1.0.0",
# # # # #         "endpoints": [
# # # # #             "/api/chatbot - Health chatbot",
# # # # #             "/api/health/symptoms - Symptom analysis",
# # # # #             "/api/health/symptoms/quick - Quick symptom check",
# # # # #             "/docs - API documentation"
# # # # #         ]
# # # # #     }

# # # # # if __name__ == "__main__":
# # # # #     import uvicorn
# # # # #     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# # # # # Open ai........................................................................


# # # # import os
# # # # import json
# # # # import asyncio
# # # # from datetime import datetime
# # # # from fastapi import FastAPI, HTTPException
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from pydantic import BaseModel
# # # # from dotenv import load_dotenv
# # # # from typing import Optional, Dict, List, Any
# # # # import logging
# # # # from contextlib import asynccontextmanager

# # # # # Import OpenAI Agents SDK components
# # # # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

# # # # # Configure logging
# # # # logging.basicConfig(level=logging.INFO)
# # # # logger = logging.getLogger(__name__)

# # # # # Load environment variables from .env file
# # # # load_dotenv()

# # # # # Global variable for chat history
# # # # chat_history: Dict[str, List[dict]] = {}

# # # # @asynccontextmanager
# # # # async def lifespan(app: FastAPI):
# # # #     # Startup
# # # #     logger.info("Starting Medicura-AI Health Assistant")
# # # #     yield
# # # #     # Shutdown
# # # #     logger.info("Shutting down Medicura-AI Health Assistant")

# # # # app = FastAPI(title="Medicura-AI Health Assistant", 
# # # #               description="AI-powered health assistant for symptom analysis and medical queries",
# # # #               version="1.0.0",
# # # #               lifespan=lifespan)

# # # # # --- CORS Configuration ---
# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
# # # #     allow_credentials=True,
# # # #     allow_methods=["*"],
# # # #     allow_headers=["*"],
# # # # )

# # # # # --- Environment Variable Loading ---
# # # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # # if not GEMINI_API_KEY:
# # # #     logger.error("GEMINI_API_KEY not found in environment variables")

# # # # # --- AI Agent Initialization (Gemini via OpenAI SDK compatibility) ---
# # # # external_client = AsyncOpenAI(
# # # #     api_key=GEMINI_API_KEY,
# # # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # # # )

# # # # model = OpenAIChatCompletionsModel(
# # # #     openai_client=external_client,
# # # #     model="gemini-2.0-flash",
# # # # )

# # # # config = RunConfig(
# # # #     model=model,
# # # #     model_provider=external_client,
# # # #     tracing_disabled=True,
# # # # )

# # # # # Define specialized agents for different medical tasks
# # # # class MedicalAgent:
# # # #     """Base class for medical specialized agents"""
    
# # # #     def __init__(self, specialty: str, system_prompt: str):
# # # #         self.specialty = specialty
# # # #         self.system_prompt = system_prompt
# # # #         self.agent = Agent(
# # # #             name=f"Medicura{specialty}Agent",
# # # #             instructions=system_prompt,
# # # #             model=model,
# # # #         )
    
# # # #     async def query(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# # # #         """Execute a query with the specialized agent"""
# # # #         try:
# # # #             full_prompt = self._build_prompt(prompt, context)
# # # #             result = await Runner.run(self.agent, full_prompt, config=config)
# # # #             return self._parse_response(result.final_output)
# # # #         except Exception as e:
# # # #             logger.error(f"Error in {self.specialty} agent: {str(e)}")
# # # #             return self._get_fallback_response()
    
# # # #     def _build_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
# # # #         """Build the complete prompt with context"""
# # # #         base_prompt = f"{self.system_prompt}\n\nUser Query: {prompt}"
# # # #         if context:
# # # #             context_str = json.dumps(context, indent=2)
# # # #             base_prompt += f"\n\nContext: {context_str}"
# # # #         return base_prompt + "\n\nPlease respond with valid JSON only."
    
# # # #     def _parse_response(self, response: str) -> Dict[str, Any]:
# # # #         """Parse the agent response"""
# # # #         try:
# # # #             # Clean the response
# # # #             cleaned = response.strip()
# # # #             if cleaned.startswith("```json"):
# # # #                 cleaned = cleaned[7:].rsplit("```", 1)[0].strip()
# # # #             elif cleaned.startswith("```"):
# # # #                 cleaned = cleaned[3:].rsplit("```", 1)[0].strip()
            
# # # #             return json.loads(cleaned)
# # # #         except json.JSONDecodeError:
# # # #             logger.warning(f"Failed to parse JSON response: {response[:200]}...")
# # # #             return self._get_fallback_response()
    
# # # #     def _get_fallback_response(self) -> Dict[str, Any]:
# # # #         """Get a fallback response when the agent fails"""
# # # #         return {
# # # #             "response": "I apologize, but I'm experiencing technical difficulties. Please consult a healthcare professional for accurate medical advice.",
# # # #             "suggestions": ["Try rephrasing your question", "Contact a healthcare provider"],
# # # #             "disclaimer": "This information may not be accurate. Always consult healthcare professionals for medical advice."
# # # #         }

# # # # # Initialize specialized agents
# # # # symptom_analyzer_agent = MedicalAgent(
# # # #     "SymptomAnalysis",
# # # #     """You are Medicura-AI Symptom Analysis Specialist. Analyze symptoms and provide professional medical guidance.

# # # # GUIDELINES:
# # # # 1. Be specific, evidence-based, and professional
# # # # 2. List potential conditions with likelihood percentages
# # # # 3. Provide clear recommendations for when to seek medical help
# # # # 4. Include safe home remedies when appropriate
# # # # 5. Always emphasize the need for professional diagnosis

# # # # RESPONSE FORMAT (JSON):
# # # # {
# # # #     "analysis": "detailed analysis",
# # # #     "potential_conditions": [{"condition": "name", "likelihood": "low/medium/high", "description": "brief explanation"}],
# # # #     "recommended_actions": ["action1", "action2"],
# # # #     "when_to_seek_help": ["situation1", "situation2"],
# # # #     "home_remedies": ["remedy1", "remedy2"],
# # # #     "disclaimer": "consultation disclaimer"
# # # # }"""
# # # # )

# # # # drug_interaction_agent = MedicalAgent(
# # # #     "DrugInteraction",
# # # #     """You are Medicura-AI Drug Interaction Specialist. Analyze medication interactions with precision.

# # # # GUIDELINES:
# # # # 1. Identify exact medication names (brand and generic)
# # # # 2. Provide interaction severity (none, mild, moderate, severe)
# # # # 3. Include mechanism of interaction when known
# # # # 4. Suggest alternatives when appropriate
# # # # 5. Reference known pharmacological interactions

# # # # RESPONSE FORMAT (JSON):
# # # # {
# # # #     "interactions": [{
# # # #         "medications": ["med1", "med2"],
# # # #         "severity": "none/mild/moderate/severe",
# # # #         "description": "interaction details",
# # # #         "recommendation": "specific advice",
# # # #         "mechanism": "pharmacological mechanism if known"
# # # #     }],
# # # #     "recommendations": ["rec1", "rec2"],
# # # #     "alternative_options": ["option1", "option2"],
# # # #     "general_advice": ["advice1", "advice2"],
# # # #     "disclaimer": "consultation disclaimer"
# # # # }"""
# # # # )

# # # # general_health_agent = MedicalAgent(
# # # #     "GeneralHealth",
# # # #     """You are Medicura-AI General Health Consultant. Provide accurate health information.

# # # # GUIDELINES:
# # # # 1. Be comprehensive and evidence-based
# # # # 2. Cite reputable sources when possible
# # # # 3. Provide practical, actionable advice
# # # # 4. Distinguish between facts and recommendations
# # # # 5. Always include appropriate cautions

# # # # RESPONSE FORMAT (JSON):
# # # # {
# # # #     "response": "detailed answer",
# # # #     "key_points": ["point1", "point2", "point3"],
# # # #     "suggestions": ["suggestion1", "suggestion2"],
# # # #     "references": ["source1", "source2"] if available,
# # # #     "disclaimer": "consultation disclaimer"
# # # # }"""
# # # # )

# # # # # Add to your existing MedicalAgent class or create a new one
# # # # medical_term_agent = MedicalAgent(
# # # #     "MedicalTerm",
# # # #     """You are Medicura-AI Medical Terminology Specialist. Explain medical terms in simple, understandable language.

# # # # GUIDELINES:
# # # # 1. Provide clear, concise explanations suitable for patients
# # # # 2. Include pronunciation guidance when helpful
# # # # 3. List key points about the term
# # # # 4. Suggest related medical terms
# # # # 5. Explain in the language requested by the user

# # # # RESPONSE FORMAT (JSON):
# # # # {
# # # #     "response": "detailed explanation in simple language",
# # # #     "key_points": ["point1", "point2", "point3"],
# # # #     "related_terms": ["term1", "term2", "term3"],
# # # #     "pronunciation": "phonetic pronunciation if helpful",
# # # #     "disclaimer": "consultation disclaimer"
# # # # }"""
# # # # )

# # # # # Add to your existing MedicalAgent class or create a new one
# # # # report_analyzer_agent = MedicalAgent(
# # # #     "ReportAnalysis",
# # # #     """You are Medicura-AI Medical Report Analyst. Analyze and summarize medical reports with clinical accuracy.

# # # # GUIDELINES:
# # # # 1. Provide concise, accurate summaries of medical reports
# # # # 2. Highlight abnormal findings and critical values
# # # # 3. Suggest appropriate follow-up actions
# # # # 4. Use layman's terms when possible for patient understanding
# # # # 5. Maintain clinical accuracy and professionalism

# # # # RESPONSE FORMAT (JSON):
# # # # {
# # # #     "summary": "comprehensive summary of the report",
# # # #     "key_findings": ["finding1", "finding2", "finding3"],
# # # #     "recommendations": ["recommendation1", "recommendation2"],
# # # #     "next_steps": ["step1", "step2", "step3"],
# # # #     "disclaimer": "consultation disclaimer"
# # # # }"""
# # # # )

# # # # # Pydantic models for request body
# # # # class SymptomInput(BaseModel):
# # # #     symptoms: List[str]
# # # #     duration: Optional[str] = None
# # # #     severity: Optional[str] = None
# # # #     age: Optional[int] = None
# # # #     gender: Optional[str] = None
# # # #     existing_conditions: Optional[List[str]] = None
# # # #     medications: Optional[List[str]] = None

# # # # class DrugInteractionInput(BaseModel):
# # # #     medications: List[str]
# # # #     age: Optional[int] = None
# # # #     gender: Optional[str] = None
# # # #     existing_conditions: Optional[List[str]] = None
# # # #     other_medications: Optional[List[str]] = None

# # # # class MedicalTermInput(BaseModel):
# # # #     term: str

# # # # class HealthQueryInput(BaseModel):
# # # #     query: str
# # # #     context: Optional[dict] = None

# # # # class ChatRequest(BaseModel):
# # # #     message: str
# # # #     session_id: Optional[str] = None
# # # #     context: Optional[dict] = None

# # # # class ClearSessionRequest(BaseModel):
# # # #     session_id: Optional[str] = None


# # # # class MedicalTermInput(BaseModel):
# # # #     term: str
# # # #     language: Optional[str] = "en"    


# # # # class ReportSummaryInput(BaseModel):
# # # #     text: Optional[str] = None
# # # #     language: Optional[str] = "en"

# # # # # API endpoints
# # # # @app.post("/api/health/symptoms")
# # # # async def analyze_symptoms(input_data: SymptomInput):
# # # #     """Analyze symptoms using specialized agent"""
# # # #     try:
# # # #         context = {
# # # #             "symptoms": input_data.symptoms,
# # # #             "duration": input_data.duration,
# # # #             "severity": input_data.severity,
# # # #             "age": input_data.age,
# # # #             "gender": input_data.gender,
# # # #             "existing_conditions": input_data.existing_conditions,
# # # #             "medications": input_data.medications
# # # #         }
        
# # # #         prompt = f"Analyze these symptoms: {', '.join(input_data.symptoms)}"
# # # #         result = await symptom_analyzer_agent.query(prompt, context)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Error analyzing symptoms: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/health/drug-interactions")
# # # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # # #     """Check drug interactions using specialized agent"""
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
# # # #         result = await drug_interaction_agent.query(prompt, context)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Error checking drug interactions: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/health/drug-info")
# # # # async def get_drug_information(medication: str):
# # # #     """Get drug information using specialized agent"""
# # # #     try:
# # # #         if not medication:
# # # #             raise HTTPException(status_code=400, detail="Medication name is required")
        
# # # #         prompt = f"Provide information about: {medication}"
# # # #         result = await drug_interaction_agent.query(prompt)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Error getting drug information: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")


# # # # @app.post("/api/health/medical-term")
# # # # async def explain_medical_term(input_data: MedicalTermInput):
# # # #     """Explain a medical term using specialized agent"""
# # # #     try:
# # # #         if not input_data.term:
# # # #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# # # #         prompt = f"Explain the medical term: {input_data.term}"
# # # #         if input_data.language and input_data.language != "en":
# # # #             prompt += f"\nPlease provide explanation in {input_data.language} language"
        
# # # #         result = await general_health_agent.query(prompt)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Error explaining medical term: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")
# # # # @app.post("/api/health/report-summarize")
# # # # async def summarize_medical_report(input_data: ReportSummaryInput):
# # # #     """Summarize medical reports using specialized agent"""
# # # #     try:
# # # #         if not input_data.text:
# # # #             raise HTTPException(status_code=400, detail="Report text is required")
        
# # # #         prompt = f"""
# # # #         Analyze and summarize this medical report:

# # # #         {input_data.text}

# # # #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# # # #         Focus on key findings, recommendations, and next steps.
# # # #         """

# # # #         result = await report_analyzer_agent.query(prompt)
# # # #         return result
        
# # # #     except Exception as e:
# # # #         logger.error(f"Error summarizing medical report: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")
    
# # # # @app.post("/api/chatbot")
# # # # async def chatbot(request: ChatRequest):
# # # #     """Handle general health queries"""
# # # #     try:
# # # #         session_id = request.session_id or "default_session"
        
# # # #         if session_id not in chat_history:
# # # #             chat_history[session_id] = []

# # # #         # Use appropriate agent based on query content
# # # #         query_lower = request.message.lower()
# # # #         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea']):
# # # #             agent = symptom_analyzer_agent
# # # #         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction']):
# # # #             agent = drug_interaction_agent
# # # #         else:
# # # #             agent = general_health_agent

# # # #         result = await agent.query(request.message, request.context)

# # # #         # Update chat history
# # # #         chat_history[session_id].extend([
# # # #             {"role": "user", "content": request.message},
# # # #             {"role": "assistant", "content": result.get("response", "I apologize, I couldn't process your request.")}
# # # #         ])
        
# # # #         # Keep last 10 messages
# # # #         chat_history[session_id] = chat_history[session_id][-10:]

# # # #         return {
# # # #             "response": result.get("response", ""),
# # # #             "suggestions": result.get("suggestions", []),
# # # #             "disclaimer": result.get("disclaimer", ""),
# # # #             "timestamp": datetime.now().isoformat(),
# # # #             "session_id": session_id
# # # #         }

# # # #     except Exception as e:
# # # #         logger.error(f"Error in chatbot: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # # @app.post("/api/chatbot/session/clear")
# # # # async def clear_session(request: ClearSessionRequest):
# # # #     """Clear chatbot session history"""
# # # #     try:
# # # #         session_id = request.session_id or "default_session"
# # # #         if session_id in chat_history:
# # # #             del chat_history[session_id]
# # # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # # #     except Exception as e:
# # # #         logger.error(f"Error clearing session: {str(e)}")
# # # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # # @app.get("/health")
# # # # async def health_check():
# # # #     """Health check endpoint"""
# # # #     return {
# # # #         "status": "healthy",
# # # #         "timestamp": datetime.now().isoformat(),
# # # #         "version": "1.0.0",
# # # #         "agents_available": True
# # # #     }

# # # # @app.get("/")
# # # # async def root():
# # # #     """Root endpoint"""
# # # #     return {
# # # #         "message": "Medicura-AI Health Assistant API",
# # # #         "version": "1.0.0",
# # # #         "endpoints": [
# # # #             "/api/chatbot - Health chatbot",
# # # #             "/api/health/symptoms - Symptom analysis",
# # # #             "/api/health/drug-interactions - Drug interaction checking",
# # # #             "/api/health/drug-info - Medication information",
# # # #             "/docs - API documentation"
# # # #         ]
# # # #     }

# # # # if __name__ == "__main__":
# # # #     import uvicorn
# # # #     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




# # # # implimentation>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # # import os
# # # import json
# # # import asyncio
# # # from datetime import datetime
# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel
# # # from dotenv import load_dotenv
# # # from typing import Optional, Dict, List, Any
# # # import logging
# # # from contextlib import asynccontextmanager
# # # import uuid
# # # import httpx
# # # import re

# # # # Import OpenAI Agents SDK components
# # # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # # # Configure logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # # Load environment variables from .env file
# # # load_dotenv()

# # # # Global variable for chat history
# # # chat_history: Dict[str, List[dict]] = {}

# # # @asynccontextmanager
# # # async def lifespan(app: FastAPI):
# # #     # Startup
# # #     logger.info("Starting Medicura-AI Health Assistant")
# # #     yield
# # #     # Shutdown
# # #     logger.info("Shutting down Medicura-AI Health Assistant")

# # # app = FastAPI(title="Medicura-AI Health Assistant", 
# # #               description="AI-powered health assistant for symptom analysis and medical queries",
# # #               version="2.1.0",
# # #               lifespan=lifespan)

# # # # --- CORS Configuration ---
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
# # #     allow_credentials=True,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )

# # # # --- Environment Variable Loading ---
# # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # if not GEMINI_API_KEY:
# # #     logger.error("GEMINI_API_KEY not found in environment variables")
# # #     raise ValueError("GEMINI_API_KEY environment variable is required")

# # # # --- AI Agent Initialization ---
# # # external_client = AsyncOpenAI(
# # #     api_key=GEMINI_API_KEY,
# # #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# # #     http_client=httpx.AsyncClient(timeout=60.0)
# # # )

# # # # Configure model with CORRECT ModelSettings
# # # model = OpenAIChatCompletionsModel(
# # #     model="gemini-2.0-flash",
# # #     openai_client=external_client,
# # # )

# # # # Create ModelSettings for better thinking
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

# # # # Initialize specialized agents with BETTER instructions
# # # symptom_analyzer_agent = Agent(
# # #     name="MedicuraSymptomAnalysisAgent",
# # #     instructions="""You are a medical AI specialist. Analyze symptoms thoroughly and provide:
# # # - Analysis of symptoms
# # # - Possible conditions
# # # - Recommended actions
# # # - When to seek help
# # # - Home remedies if appropriate
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY, no other text.""",
# # #     model=model,
# # # )

# # # drug_interaction_agent = Agent(
# # #     name="MedicuraDrugInteractionAgent",
# # #     instructions="""You are a pharmacology AI expert. Analyze drug interactions and provide:
# # # - Interaction details
# # # - Severity levels
# # # - Recommendations
# # # - Alternative options
# # # - General advice
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY, no other text.""",
# # #     model=model,
# # # )

# # # general_health_agent = Agent(
# # #     name="MedicuraGeneralHealthAgent",
# # #     instructions="""You are a health AI consultant. Provide accurate information and:
# # # - Detailed response
# # # - Key points
# # # - Suggestions
# # # - References if available
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY, no other text.""",
# # #     model=model,
# # # )

# # # medical_term_agent = Agent(
# # #     name="MedicuraMedicalTermAgent",
# # #     instructions="""You are a medical terminology specialist. Explain terms and provide:
# # # - Term definition
# # # - Key points
# # # - Pronunciation
# # # - Related terms
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY, no other text.""",
# # #     model=model,
# # # )

# # # report_analyzer_agent = Agent(
# # #     name="MedicuraReportAnalysisAgent",
# # #     instructions="""You are a medical report analyst. Summarize reports and provide:
# # # - Comprehensive summary
# # # - Key findings
# # # - Recommendations
# # # - Next steps
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY, no other text.""",
# # #     model=model,
# # # )

# # # def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
# # #     """Extract JSON from agent response with multiple fallback methods"""
# # #     try:
# # #         # Method 1: Try direct JSON parsing
# # #         try:
# # #             return json.loads(response.strip())
# # #         except json.JSONDecodeError:
# # #             pass
        
# # #         # Method 2: Extract from code blocks
# # #         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
# # #         if json_match:
# # #             return json.loads(json_match.group(1))
        
# # #         # Method 3: Find first JSON object
# # #         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
# # #         if brace_match:
# # #             return json.loads(brace_match.group(0))
            
# # #         # Method 4: If all else fails, create structured response from text
# # #         return {
# # #             "analysis": response,
# # #             "key_points": ["Important medical information provided", "Professional consultation recommended"],
# # #             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
# # #             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice."
# # #         }
        
# # #     except Exception as e:
# # #         logger.warning(f"JSON extraction failed: {str(e)}")
# # #         return None

# # # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# # #     """Run agent with enhanced thinking and robust error handling"""
# # #     try:
# # #         # Build thinking-enhanced prompt
# # #         thinking_prompt = f"""
# # #         USER QUERY: {prompt}
# # #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
        
# # #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# # #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# # #         """
        
# # #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# # #         # Extract and parse JSON from response
# # #         parsed_response = extract_json_from_response(result.final_output)
        
# # #         if parsed_response:
# # #             # Add metadata
# # #             parsed_response.update({
# # #                 "timestamp": datetime.now().isoformat(),
# # #                 "success": True,
# # #                 "thinking_applied": True
# # #             })
# # #             return parsed_response
# # #         else:
# # #             # Fallback if JSON extraction fails
# # #             return create_intelligent_response(result.final_output, prompt)
        
# # #     except Exception as e:
# # #         logger.error(f"Agent error: {str(e)}")
# # #         return create_intelligent_response(f"Analysis of: {prompt}")

# # # def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
# # #     """Create a well-structured response from text"""
# # #     return {
# # #         "analysis": response_text if response_text else f"Comprehensive analysis of: {original_query}",
# # #         "key_points": [
# # #             "Professional medical consultation is recommended",
# # #             "Individual health circumstances vary",
# # #             "Accurate diagnosis requires proper medical examination"
# # #         ],
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
# # #         "timestamp": datetime.now().isoformat(),
# # #         "success": True,
# # #         "thinking_applied": True
# # #     }

# # # # Pydantic models
# # # class ChatRequest(BaseModel):
# # #     message: str
# # #     session_id: Optional[str] = None
# # #     context: Optional[dict] = None
    

# # # class DrugInteractionInput(BaseModel):
# # #     medications: List[str]
# # #     age: Optional[int] = None
# # #     gender: Optional[str] = None
# # #     existing_conditions: Optional[List[str]] = None
# # #     other_medications: Optional[List[str]] = None

# # # class MedicalTermInput(BaseModel):
# # #     term: str
# # #     language: Optional[str] = "en"

# # # class ReportTextInput(BaseModel):
# # #     text: str
# # #     language: Optional[str] = "en"

# # # class ClearSessionRequest(BaseModel):
# # #     session_id: Optional[str] = None

# # # # API endpoints
# # # @app.post("/api/chatbot")
# # # async def chatbot(request: ChatRequest):
# # #     """Main chatbot endpoint with intelligent thinking"""
# # #     try:
# # #         session_id = request.session_id or str(uuid.uuid4())
        
# # #         if session_id not in chat_history:
# # #             chat_history[session_id] = []

# # #         # Use appropriate agent
# # #         query_lower = request.message.lower()
        
# # #         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea', 'ache', 'hurt']):
# # #             agent = symptom_analyzer_agent
# # #         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction', 'side effect', 'ibuprofen', 'glutathion']):
# # #             agent = drug_interaction_agent
# # #         elif any(term in query_lower for term in ['what is', 'explain', 'define', 'meaning of']):
# # #             agent = medical_term_agent
# # #         elif any(term in query_lower for term in ['report', 'result', 'test', 'lab', 'x-ray', 'summary']):
# # #             agent = report_analyzer_agent
# # #         else:
# # #             agent = general_health_agent

# # #         # Get intelligent response
# # #         result = await run_agent_with_thinking(agent, request.message, request.context)
# # #         # print(result.final_output)
        

# # #         # Update chat history
# # #         chat_history[session_id].extend([
# # #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# # #             {"role": "assistant", "content": result.get("analysis", result.get("response", "")), "timestamp": datetime.now().isoformat()}
# # #         ])
        
# # #         # Keep last 20 messages
# # #         chat_history[session_id] = chat_history[session_id][-20:]

# # #         return result

# # #     except Exception as e:
# # #         logger.error(f"Chatbot error: {str(e)}")
# # #         return create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")

# # # @app.post("/api/health/drug-interactions")
# # # async def check_drug_interactions(input_data: DrugInteractionInput):
# # #     """Check drug interactions with thorough analysis"""
# # #     try:
# # #         if not input_data.medications or len(input_data.medications) == 0:
# # #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# # #         context = {
# # #             "medications": input_data.medications,
# # #             "age": input_data.age,
# # #             "gender": input_data.gender,
# # #             "existing_conditions": input_data.existing_conditions,
# # #             "other_medications": input_data.other_medications
# # #         }
        
# # #         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
# # #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Drug interaction error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/health/medical-term")
# # # async def explain_medical_term(input_data: MedicalTermInput):
# # #     """Explain medical terms with clarity"""
# # #     try:
# # #         if not input_data.term:
# # #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# # #         prompt = f"Explain the medical term: {input_data.term}"
# # #         if input_data.language and input_data.language != "en":
# # #             prompt += f" in {input_data.language} language"
        
# # #         result = await run_agent_with_thinking(medical_term_agent, prompt)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Medical term error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/health/report-summarize")
# # # async def summarize_medical_report(input_data: ReportTextInput):
# # #     """Summarize medical reports with intelligent analysis"""
# # #     try:
# # #         if not input_data.text:
# # #             raise HTTPException(status_code=400, detail="Report text is required")
        
# # #         prompt = f"""
# # #         Analyze and summarize this medical report:

# # #         {input_data.text}

# # #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# # #         Focus on key findings, recommendations, and next steps.
# # #         """

# # #         result = await run_agent_with_thinking(report_analyzer_agent, prompt)
# # #         return result
        
# # #     except Exception as e:
# # #         logger.error(f"Report summary error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # # @app.post("/api/chatbot/session/clear")
# # # async def clear_session(request: ClearSessionRequest):
# # #     """Clear chatbot session history"""
# # #     try:
# # #         session_id = request.session_id or "default_session"
# # #         if session_id in chat_history:
# # #             del chat_history[session_id]
# # #         return {"message": "Session cleared successfully", "session_id": session_id}
# # #     except Exception as e:
# # #         logger.error(f"Clear session error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # # @app.get("/health")
# # # async def health_check():
# # #     """Health check endpoint"""
# # #     return {
# # #         "status": "healthy",
# # #         "timestamp": datetime.now().isoformat(),
# # #         "version": "2.1.0",
# # #         "agents_available": True,
# # #         "thinking_enabled": True
# # #     }

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)



# # import os
# # import json
# # import asyncio
# # from datetime import datetime
# # from fastapi import FastAPI, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse
# # from pydantic import BaseModel, Field
# # from dotenv import load_dotenv
# # from typing import Optional, Dict, List, Any
# # import logging
# # from contextlib import asynccontextmanager
# # import uuid
# # import httpx
# # import re

# # # Import OpenAI Agents SDK components
# # from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Load environment variables from .env file
# # load_dotenv()

# # # Global variable for chat history
# # chat_history: Dict[str, List[dict]] = {}

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     # Startup
# #     logger.info("Starting Medicura-AI Health Assistant")
# #     yield
# #     # Shutdown
# #     logger.info("Shutting down Medicura-AI Health Assistant")

# # app = FastAPI(
# #     title="Medicura-AI Health Assistant", 
# #     description="AI-powered health assistant for symptom analysis and medical queries",
# #     version="2.1.0",
# #     lifespan=lifespan,
# #     docs_url="/api/docs",
# #     redoc_url="/api/redoc"
# # )

# # # --- CORS Configuration ---
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # --- Environment Variable Loading ---
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # if not GEMINI_API_KEY:
# #     logger.error("GEMINI_API_KEY not found in environment variables")
# #     raise ValueError("GEMINI_API_KEY environment variable is required")

# # # --- AI Agent Initialization ---
# # external_client = AsyncOpenAI(
# #     api_key=GEMINI_API_KEY,
# #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# #     http_client=httpx.AsyncClient(timeout=60.0)
# # )

# # # Configure model with CORRECT ModelSettings
# # model = OpenAIChatCompletionsModel(
# #     model="gemini-2.0-flash",
# #     openai_client=external_client,
# # )

# # # Create ModelSettings for better thinking
# # model_settings = ModelSettings(
# #     temperature=0.7,
# #     top_p=0.9,
# #     max_tokens=2048,
# # )

# # config = RunConfig(
# #     model=model,
# #     model_provider=external_client,
# #     model_settings=model_settings,
# #     tracing_disabled=True,
# # )

# # # # Initialize specialized agents with BETTER instructions
# # # symptom_analyzer_agent = Agent(
# # #     name="MedicuraSymptomAnalysisAgent",
# # #     instructions="""You are a medical AI specialist. Analyze symptoms thoroughly and provide:
# # # - Analysis of symptoms
# # # - Possible conditions
# # # - Recommended actions
# # # - When to seek help
# # # - Home remedies if appropriate
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, when_to_seek_help, disclaimer, type.
# # # NO OTHER TEXT.""",
# # #     model=model,
# # # )
# # symptom_analyzer_agent = Agent(
# #     name="MedicuraSymptomAnalysisAgent",
# #     instructions="""
# # You are MedicuraSymptomAnalysisAgent, a cautious medical AI for triage and health education (not a doctor).

# # GOAL
# # - Analyze user-reported symptoms and provide safe, concise triage guidance.

# # OUTPUT (STRICT)
# # Return PURE JSON with exactly these keys:
# # - summary            (1–2 lines, user-facing)
# # - detailed_analysis  (bullet-like text; reasoning, missing info, serious vs common causes)
# # - recommendations    (numbered, actionable steps)
# # - when_to_seek_help  (clear triggers in plain language)
# # - disclaimer         (fixed safety line)
# # - type               (one of: "emergency", "urgent", "non-urgent", "self-care")
# # No extra keys, no markdown, no code fences, no nulls (use empty string if unsure).

# # LANGUAGE & TONE
# # - Match the user’s language (English or Roman Urdu/Urdu). Keep sentences short, clear, and empathetic. No emojis.

# # SAFETY FIRST — RED FLAGS (if any present, default to safety)
# # Set type="emergency" and prioritize urgent care if symptoms include ANY of:
# # - Chest pain WITH shortness of breath, sweating, nausea, fainting, or pain radiating to arm/jaw/back
# # - Signs of stroke (FACE: Face droop, Arm weakness, Speech difficulties; sudden vision loss; severe imbalance)
# # - “Worst-ever” sudden headache, or headache with fever, neck stiffness, confusion, fainting
# # - Severe allergic reaction (trouble breathing, swelling of face/tongue/throat, widespread hives, dizziness)
# # - Severe abdominal pain with guarding, persistent vomiting, blood in vomit/stool, or black/tarry stools
# # - High fever >103°F (39.4°C), fever with lethargy in a child, or any fever in infants <3 months
# # - Pregnancy with heavy bleeding, severe abdominal pain, severe headache/vision changes, or reduced fetal movement
# # - Severe dehydration or inability to keep fluids down
# # - Suicidal thoughts or self-harm intent
# # - Major trauma, head injury, large burns, suspected poisoning
# # - Diabetes with vomiting, confusion, fruity breath, very high sugars

# # TRIAGE LEVELS
# # - "emergency": Tell user to call local emergency services NOW (Pakistan example: 1122) or go to nearest ED; do not drive self.
# # - "urgent": See a clinician within 24 hours.
# # - "non-urgent": Primary care in 2–3 days.
# # - "self-care": Manage at home with precautions.

# # REASONING RULES
# # - Use only provided info; note “Missing info: …” in detailed_analysis and stay conservative.
# # - Group differentials as Serious vs Common; never state a diagnosis—use “possible”, “consider”.
# # - Factor onset, duration, severity, triggers, relieving/aggravating factors.
# # - Higher risk: pregnancy, age <5 years or >65 years, or chronic heart/lung/kidney/liver disease, diabetes—upgrade triage when appropriate.

# # MEDICATION GUIDANCE (OTC only)
# # - Mention generics (e.g., paracetamol/acetaminophen, ibuprofen) with “follow label dosing”.
# # - Do NOT provide exact doses unless age/weight is given and safe; never exceed maximums.
# # - Safety cautions: No aspirin for children/teens with viral illness; avoid NSAIDs in pregnancy/ulcer/kidney disease.

# # EMERGENCY WORDING (must be first in recommendations when type="emergency")
# # - "Call emergency services (e.g., 1122) or go to the nearest emergency department now. Do not drive yourself."

# # DISCLAIMER (use this exact text)
# # - "This information is educational and not a diagnosis. For medical advice and treatment, consult a licensed clinician."
# # """,
# #     model=model,
# # )


# # # drug_interaction_agent = Agent(
# # #     name="MedicuraDrugInteractionAgent",
# # #     instructions="""You are a pharmacology AI expert. Analyze drug interactions and provide:
# # # - Interaction details
# # # - Severity levels
# # # - Recommendations
# # # - Alternative options
# # # - General advice
# # # - Always include disclaimer

# # # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, severity, alternative_options, disclaimer, type.
# # # NO OTHER TEXT.""",
# # #     model=model,
# # # )

# # drug_interaction_agent = Agent(
# #     name="MedicuraDrugInteractionAgent",
# #     instructions="""
# # You are a highly knowledgeable pharmacology AI assistant specializing in **drug interactions and medication safety**.  
# # Your task is to analyze the provided list of medications and generate a structured professional report.  

# # Follow these rules strictly:  
# # - Always base your response on **evidence-based pharmacology knowledge**.  
# # - If limited or no reliable information is available, clearly state that.  
# # - Keep answers **concise but medically informative**.  
# # - DO NOT include any text outside of the required JSON fields.  

# # RETURN PURE JSON ONLY with these exact fields:  
# # {
# #   "summary": "One-sentence overview of the interaction findings",
# #   "detailed_analysis": "Concise explanation of potential interactions, mechanisms (if known), and effects",
# #   "recommendations": "Practical next steps for safe medication use",
# #   "severity": "One of: None, Mild, Moderate, Severe, Unknown",
# #   "alternative_options": "List possible safer alternatives if applicable, otherwise 'None'",
# #   "disclaimer": "This is not medical advice. Always consult a qualified healthcare professional.",
# #   "type": "drug_interaction"
# # }

# # Rules for response:  
# # - Use **'None'** or **'Unknown'** instead of guessing.  
# # - Recommendations must always include **consulting a healthcare provider**.  
# # - Severity must be clearly one of the defined categories.  
# # """,
# #     model=model,
# # )


# # general_health_agent = Agent(
# #     name="MedicuraGeneralHealthAgent",
# #     instructions="""You are a health AI consultant. Provide accurate information and:
# # - Detailed response
# # - Key points
# # - Suggestions
# # - References if available
# # - Always include disclaimer

# # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# # NO OTHER TEXT.""",
# #     model=model,
# # )

# # medical_term_agent = Agent(
# #     name="MedicuraMedicalTermAgent",
# #     instructions="""You are a medical terminology specialist. Explain terms and provide:
# # - Term definition
# # - Key points
# # - Pronunciation
# # - Related terms
# # - Always include disclaimer

# # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# # NO OTHER TEXT.""",
# #     model=model,
# # )

# # report_analyzer_agent = Agent(
# #     name="MedicuraReportAnalysisAgent",
# #     instructions="""You are a medical report analyst. Summarize reports and provide:
# # - Comprehensive summary
# # - Key findings
# # - Recommendations
# # - Next steps
# # - Always include disclaimer

# # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_findings, next_steps, disclaimer, type.
# # NO OTHER TEXT.""",
# #     model=model,
# # )

# # about_agent = Agent(
# #     name="MedicuraAboutAgent",
# #     instructions="""You provide information about Medicura-AI Health and its creator Hadiqa Gohar. Include:
# # - Creator background and skills
# # - Features of Medicura-AI
# # - Technical capabilities
# # - Always be friendly and helpful

# # RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# # NO OTHER TEXT.""",
# #     model=model,
# # )

# # def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
# #     """Extract JSON from agent response with multiple fallback methods"""
# #     try:
# #         # Method 1: Try direct JSON parsing
# #         try:
# #             return json.loads(response.strip())
# #         except json.JSONDecodeError:
# #             pass
        
# #         # Method 2: Extract from code blocks
# #         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
# #         if json_match:
# #             return json.loads(json_match.group(1))
        
# #         # Method 3: Find first JSON object
# #         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
# #         if brace_match:
# #             return json.loads(brace_match.group(0))
            
# #         # Method 4: If all else fails, create structured response from text
# #         return {
# #             "summary": response,
# #             "detailed_analysis": "Detailed analysis based on your query",
# #             "recommendations": ["Consult with healthcare provider", "Follow medical guidance"],
# #             "disclaimer": "This information is for educational purposes. Consult healthcare professionals for medical advice.",
# #             "type": "general"
# #         }
        
# #     except Exception as e:
# #         logger.warning(f"JSON extraction failed: {str(e)}")
# #         return None

# # async def run_agent_with_thinking(agent: Agent, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
# #     """Run agent with enhanced thinking and robust error handling"""
# #     try:
# #         # Build thinking-enhanced prompt
# #         thinking_prompt = f"""
# #         USER QUERY: {prompt}
# #         CONTEXT: {json.dumps(context) if context else 'No additional context'}
        
# #         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
# #         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
# #         """
        
# #         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
# #         # Extract and parse JSON from response
# #         parsed_response = extract_json_from_response(result.final_output)
        
# #         if parsed_response:
# #             # Add metadata
# #             parsed_response.update({
# #                 "timestamp": datetime.now().isoformat(),
# #                 "success": True,
# #                 "thinking_applied": True
# #             })
# #             return parsed_response
# #         else:
# #             # Fallback if JSON extraction fails
# #             return create_intelligent_response(result.final_output, prompt)
        
# #     except Exception as e:
# #         logger.error(f"Agent error: {str(e)}")
# #         return create_intelligent_response(f"Analysis of: {prompt}")

# # def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
# #     """Create a well-structured response from text"""
# #     return {
# #         "summary": response_text if response_text else f"Comprehensive analysis of: {original_query}",
# #         "detailed_analysis": "I've analyzed your query and here's what you should know based on current medical knowledge.",
# #         "recommendations": [
# #             "Consult with a healthcare provider",
# #             "Provide complete medical history for assessment",
# #             "Follow evidence-based medical guidance"
# #         ],
# #         "when_to_seek_help": [
# #             "Immediately for severe or emergency symptoms",
# #             "Within 24-48 hours for persistent concerns",
# #             "Routinely for preventive care"
# #         ],
# #         "disclaimer": "This information is for educational purposes only. Always consult healthcare professionals for medical advice.",
# #         "type": "general",
# #         "timestamp": datetime.now().isoformat(),
# #         "success": True,
# #         "thinking_applied": True
# #     }

# # # Pydantic models
# # class ChatRequest(BaseModel):
# #     message: str = Field(..., min_length=1, max_length=1000)
# #     session_id: Optional[str] = Field(None, max_length=100)
# #     context: Optional[dict] = None

# # class DrugInteractionInput(BaseModel):
# #     medications: List[str] = Field(..., min_items=1, max_items=10)
# #     age: Optional[int] = Field(None, ge=0, le=120)
# #     gender: Optional[str] = Field(None, max_length=20)
# #     existing_conditions: Optional[List[str]] = Field(None, max_items=20)
# #     other_medications: Optional[List[str]] = Field(None, max_items=20)

# # class MedicalTermInput(BaseModel):
# #     term: str = Field(..., min_length=1, max_length=100)
# #     language: Optional[str] = Field("en", max_length=10)

# # class ReportTextInput(BaseModel):
# #     text: str = Field(..., min_length=10, max_length=10000)
# #     language: Optional[str] = Field("en", max_length=10)

# # class ClearSessionRequest(BaseModel):
# #     session_id: Optional[str] = Field(None, max_length=100)

# # # API endpoints
# # @app.post("/api/chatbot")
# # async def chatbot(request: ChatRequest):
# #     """Main chatbot endpoint with intelligent thinking"""
# #     try:
# #         session_id = request.session_id or str(uuid.uuid4())
        
# #         if session_id not in chat_history:
# #             chat_history[session_id] = []

# #         # Use appropriate agent
# #         query_lower = request.message.lower()
        
# #         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea', 'ache', 'hurt']):
# #             agent = symptom_analyzer_agent
# #         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction', 'side effect', 'ibuprofen', 'glutathion']):
# #             agent = drug_interaction_agent
# #         elif any(term in query_lower for term in ['what is', 'explain', 'define', 'meaning of']):
# #             agent = medical_term_agent
# #         elif any(term in query_lower for term in ['report', 'result', 'test', 'lab', 'x-ray', 'summary']):
# #             agent = report_analyzer_agent
# #         elif any(term in query_lower for term in ['creator', 'author', 'hadiqa', 'gohar', 'medicura about', 'who made']):
# #             agent = about_agent
# #         else:
# #             agent = general_health_agent

# #         # Get intelligent response
# #         result = await run_agent_with_thinking(agent, request.message, request.context)

# #         # Update chat history
# #         chat_history[session_id].extend([
# #             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
# #             {"role": "assistant", "content": result, "timestamp": datetime.now().isoformat()}
# #         ])
        
# #         # Keep last 20 messages
# #         chat_history[session_id] = chat_history[session_id][-20:]

# #         return result

# #     except Exception as e:
# #         logger.error(f"Chatbot error: {str(e)}")
# #         return JSONResponse(
# #             status_code=500,
# #             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
# #         )

# # @app.post("/api/health/drug-interactions")
# # async def check_drug_interactions(input_data: DrugInteractionInput):
# #     """Check drug interactions with thorough analysis"""
# #     try:
# #         if not input_data.medications or len(input_data.medications) == 0:
# #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# #         context = {
# #             "medications": input_data.medications,
# #             "age": input_data.age,
# #             "gender": input_data.gender,
# #             "existing_conditions": input_data.existing_conditions,
# #             "other_medications": input_data.other_medications
# #         }
        
# #         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
# #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"Drug interaction error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")


# # @app.post("/api/health/drug-interactions")
# # async def check_drug_interactions(input_data: DrugInteractionInput):
# #     """Check drug interactions with thorough analysis"""
# #     try:
# #         if not input_data.medications or len(input_data.medications) == 0:
# #             raise HTTPException(status_code=400, detail="At least one medication is required")
        
# #         context = {
# #             "medications": input_data.medications,
# #             "age": input_data.age,
# #             "gender": input_data.gender,
# #             "existing_conditions": input_data.existing_conditions,
# #             "other_medications": input_data.other_medications
# #         }
        
# #         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
# #         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"Drug interaction error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")


# # @app.post("/api/health/medical-term")
# # async def explain_medical_term(input_data: MedicalTermInput):
# #     """Explain medical terms with clarity"""
# #     try:
# #         if not input_data.term:
# #             raise HTTPException(status_code=400, detail="Medical term is required")
        
# #         prompt = f"Explain the medical term: {input_data.term}"
# #         if input_data.language and input_data.language != "en":
# #             prompt += f" in {input_data.language} language"
        
# #         result = await run_agent_with_thinking(medical_term_agent, prompt)
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"Medical term error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # @app.post("/api/health/report-summarize")
# # async def summarize_medical_report(input_data: ReportTextInput):
# #     """Summarize medical reports with intelligent analysis"""
# #     try:
# #         if not input_data.text:
# #             raise HTTPException(status_code=400, detail="Report text is required")
        
# #         prompt = f"""
# #         Analyze and summarize this medical report:

# #         {input_data.text}

# #         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
# #         Focus on key findings, recommendations, and next steps.
# #         """

# #         result = await run_agent_with_thinking(report_analyzer_agent, prompt)
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"Report summary error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# # @app.post("/api/chatbot/session/clear")
# # async def clear_session(request: ClearSessionRequest):
# #     """Clear chatbot session history"""
# #     try:
# #         session_id = request.session_id or "default_session"
# #         if session_id in chat_history:
# #             del chat_history[session_id]
# #         return {"message": "Session cleared successfully", "session_id": session_id}
# #     except Exception as e:
# #         logger.error(f"Clear session error: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Failed to clear session")

# # @app.get("/health")
# # async def health_check():
# #     """Health check endpoint"""
# #     return {
# #         "status": "healthy",
# #         "timestamp": datetime.now().isoformat(),
# #         "version": "2.1.0",
# #         "agents_available": True,
# #         "thinking_enabled": True
# #     }

# # @app.get("/api/chatbot/sessions")
# # async def get_sessions():
# #     """Get active session count (for monitoring)"""
# #     return {
# #         "active_sessions": len(chat_history),
# #         "total_messages": sum(len(messages) for messages in chat_history.values())
# #     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)



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

# # Import OpenAI Agents SDK components
# from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # TiDB Configuration
# DB_CONFIG = {
#     "host": os.getenv("TIDB_HOST"),
#     "port": 4000,
#     "user": os.getenv("TIDB_USERNAME"),
#     "password": os.getenv("TIDB_PASSWORD"),
#     "database": os.getenv("TIDB_DATABASE"),
#     "charset": 'utf8mb4',
# }

# CA_PATH = os.getenv("CA_PATH")  # Optional: Path to CA certificate for SSL
# if CA_PATH:
#     DB_CONFIG["ssl_verify_cert"] = True
#     DB_CONFIG["ssl_verify_identity"] = True
#     DB_CONFIG["ssl_ca"] = CA_PATH

# def get_db():
#     return pymysql.connect(**DB_CONFIG)

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("Starting Medicura-AI Health Assistant")
#     logger.info("Connecting to TiDB...")
#     conn = get_db()
#     cur = conn.cursor()
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS chat_sessions (
#         session_id VARCHAR(100) PRIMARY KEY,
#         history JSON NOT NULL,
#         last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#     )
#     """)
#     conn.commit()
#     cur.close()
#     conn.close()
#     logger.info("TiDB connected and table ready.")
#     yield
#     # Shutdown
#     logger.info("Shutting down Medicura-AI Health Assistant")

# app = FastAPI(
#     title="Medicura-AI Health Assistant", 
#     description="AI-powered health assistant for symptom analysis and medical queries",
#     version="2.1.0",
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc"
# )

# # --- CORS Configuration ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Environment Variable Loading ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logger.error("GEMINI_API_KEY not found in environment variables")
#     raise ValueError("GEMINI_API_KEY environment variable is required")

# # --- AI Agent Initialization ---
# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     http_client=httpx.AsyncClient(timeout=60.0)
# )

# # Configure model with CORRECT ModelSettings
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client,
# )

# # Create ModelSettings for better thinking
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

# symptom_analyzer_agent = Agent(
#     name="MedicuraSymptomAnalysisAgent",
#     instructions="""
# You are MedicuraSymptomAnalysisAgent, a cautious medical AI for triage and health education (not a doctor).

# GOAL
# - Analyze user-reported symptoms and provide safe, concise triage guidance.

# OUTPUT (STRICT)
# Return PURE JSON with exactly these keys:
# - summary            (1–2 lines, user-facing)
# - detailed_analysis  (bullet-like text; reasoning, missing info, serious vs common causes)
# - recommendations    (numbered, actionable steps)
# - when_to_seek_help  (clear triggers in plain language)
# - disclaimer         (fixed safety line)
# - type               (one of: "emergency", "urgent", "non-urgent", "self-care")
# No extra keys, no markdown, no code fences, no nulls (use empty string if unsure).

# LANGUAGE & TONE
# - Match the user’s language (English or Roman Urdu/Urdu). Keep sentences short, clear, and empathetic. No emojis.

# SAFETY FIRST — RED FLAGS (if any present, default to safety)
# Set type="emergency" and prioritize urgent care if symptoms include ANY of:
# - Chest pain WITH shortness of breath, sweating, nausea, fainting, or pain radiating to arm/jaw/back
# - Signs of stroke (FACE: Face droop, Arm weakness, Speech difficulties; sudden vision loss; severe imbalance)
# - “Worst-ever” sudden headache, or headache with fever, neck stiffness, confusion, fainting
# - Severe allergic reaction (trouble breathing, swelling of face/tongue/throat, widespread hives, dizziness)
# - Severe abdominal pain with guarding, persistent vomiting, blood in vomit/stool, or black/tarry stools
# - High fever >103°F (39.4°C), fever with lethargy in a child, or any fever in infants <3 months
# - Pregnancy with heavy bleeding, severe abdominal pain, severe headache/vision changes, or reduced fetal movement
# - Severe dehydration or inability to keep fluids down
# - Suicidal thoughts or self-harm intent
# - Major trauma, head injury, large burns, suspected poisoning
# - Diabetes with vomiting, confusion, fruity breath, very high sugars

# TRIAGE LEVELS
# - "emergency": Tell user to call local emergency services NOW (Pakistan example: 1122) or go to nearest ED; do not drive self.
# - "urgent": See a clinician within 24 hours.
# - "non-urgent": Primary care in 2–3 days.
# - "self-care": Manage at home with precautions.

# REASONING RULES
# - Use only provided info; note “Missing info: …” in detailed_analysis and stay conservative.
# - Group differentials as Serious vs Common; never state a diagnosis—use “possible”, “consider”.
# - Factor onset, duration, severity, triggers, relieving/aggravating factors.
# - Higher risk: pregnancy, age <5 years or >65 years, or chronic heart/lung/kidney/liver disease, diabetes—upgrade triage when appropriate.

# MEDICATION GUIDANCE (OTC only)
# - Mention generics (e.g., paracetamol/acetaminophen, ibuprofen) with “follow label dosing”.
# - Do NOT provide exact doses unless age/weight is given and safe; never exceed maximums.
# - Safety cautions: No aspirin for children/teens with viral illness; avoid NSAIDs in pregnancy/ulcer/kidney disease.

# EMERGENCY WORDING (must be first in recommendations when type="emergency")
# - "Call emergency services (e.g., 1122) or go to the nearest emergency department now. Do not drive yourself."

# DISCLAIMER (use this exact text)
# - "This information is educational and not a diagnosis. For medical advice and treatment, consult a licensed clinician."
# """,
#     model=model,
# )

# drug_interaction_agent = Agent(
#     name="MedicuraDrugInteractionAgent",
#     instructions="""
# You are a highly knowledgeable pharmacology AI assistant specializing in **drug interactions and medication safety**.  
# Your task is to analyze the provided list of medications and generate a structured professional report.  

# Follow these rules strictly:  
# - Always base your response on **evidence-based pharmacology knowledge**.  
# - If limited or no reliable information is available, clearly state that.  
# - Keep answers **concise but medically informative**.  
# - DO NOT include any text outside of the required JSON fields.  

# RETURN PURE JSON ONLY with these exact fields:  
# {
#   "summary": "One-sentence overview of the interaction findings",
#   "detailed_analysis": "Concise explanation of potential interactions, mechanisms (if known), and effects",
#   "recommendations": "Practical next steps for safe medication use",
#   "severity": "One of: None, Mild, Moderate, Severe, Unknown",
#   "alternative_options": "List possible safer alternatives if applicable, otherwise 'None'",
#   "disclaimer": "This is not medical advice. Always consult a qualified healthcare professional.",
#   "type": "drug_interaction"
# }

# Rules for response:  
# - Use **'None'** or **'Unknown'** instead of guessing.  
# - Recommendations must always include **consulting a healthcare provider**.  
# - Severity must be clearly one of the defined categories.  
# """,
#     model=model,
# )

# general_health_agent = Agent(
#     name="MedicuraGeneralHealthAgent",
#     instructions="""You are a health AI consultant. Provide accurate information and:
# - Detailed response
# - Key points
# - Suggestions
# - References if available
# - Always include disclaimer

# RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# NO OTHER TEXT.""",
#     model=model,
# )

# medical_term_agent = Agent(
#     name="MedicuraMedicalTermAgent",
#     instructions="""You are a medical terminology specialist. Explain terms and provide:
# - Term definition
# - Key points
# - Pronunciation
# - Related terms
# - Always include disclaimer

# RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# NO OTHER TEXT.""",
#     model=model,
# )

# report_analyzer_agent = Agent(
#     name="MedicuraReportAnalysisAgent",
#     instructions="""You are a medical report analyst. Summarize reports and provide:
# - Comprehensive summary
# - Key findings
# - Recommendations
# - Next steps
# - Always include disclaimer

# RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_findings, next_steps, disclaimer, type.
# NO OTHER TEXT.""",
#     model=model,
# )

# about_agent = Agent(
#     name="MedicuraAboutAgent",
#     instructions="""You provide information about Medicura-AI Health and its creator Hadiqa Gohar. Include:
# - Creator background and skills
# - Features of Medicura-AI
# - Technical capabilities
# - Always be friendly and helpful

# RETURN PURE JSON ONLY with these exact fields: summary, detailed_analysis, recommendations, key_points, disclaimer, type.
# NO OTHER TEXT.""",
#     model=model,
# )

# def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
#     """Extract JSON from agent response with multiple fallback methods"""
#     try:
#         # Method 1: Try direct JSON parsing
#         try:
#             return json.loads(response.strip())
#         except json.JSONDecodeError:
#             pass
        
#         # Method 2: Extract from code blocks
#         json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group(1))
        
#         # Method 3: Find first JSON object
#         brace_match = re.search(r'\{.*\}', response, re.DOTALL)
#         if brace_match:
#             return json.loads(brace_match.group(0))
            
#         # Method 4: If all else fails, create structured response from text
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
#     """Run agent with enhanced thinking and robust error handling"""
#     try:
#         # Build thinking-enhanced prompt
#         thinking_prompt = f"""
#         USER QUERY: {prompt}
#         CONTEXT: {json.dumps(context) if context else 'No additional context'}
        
#         PLEASE PROVIDE A COMPREHENSIVE MEDICAL RESPONSE IN PURE JSON FORMAT ONLY.
#         DO NOT INCLUDE ANY OTHER TEXT OUTSIDE THE JSON.
#         """
        
#         result = await Runner.run(agent, thinking_prompt, run_config=config)
        
#         # Extract and parse JSON from response
#         parsed_response = extract_json_from_response(result.final_output)
        
#         if parsed_response:
#             # Add metadata
#             parsed_response.update({
#                 "timestamp": datetime.now().isoformat(),
#                 "success": True,
#                 "thinking_applied": True
#             })
#             return parsed_response
#         else:
#             # Fallback if JSON extraction fails
#             return create_intelligent_response(result.final_output, prompt)
        
#     except Exception as e:
#         logger.error(f"Agent error: {str(e)}")
#         return create_intelligent_response(f"Analysis of: {prompt}")

# def create_intelligent_response(response_text: str = "", original_query: str = "") -> Dict[str, Any]:
#     """Create a well-structured response from text"""
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
#     conn = get_db()
#     cur = conn.cursor()
#     cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
#     result = cur.fetchone()
#     cur.close()
#     conn.close()
#     if result:
#         return json.loads(result[0])
#     return []

# def save_history(session_id: str, history: List[dict]):
#     conn = get_db()
#     cur = conn.cursor()
#     cur.execute("""
#     INSERT INTO chat_sessions (session_id, history) VALUES (%s, %s)
#     ON DUPLICATE KEY UPDATE history = %s
#     """, (session_id, json.dumps(history), json.dumps(history)))
#     conn.commit()
#     cur.close()
#     conn.close()

# # Pydantic models
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

# # API endpoints
# @app.post("/api/chatbot")
# async def chatbot(request: ChatRequest):
#     """Main chatbot endpoint with intelligent thinking"""
#     try:
#         session_id = request.session_id or str(uuid.uuid4())
        
#         history = load_history(session_id)

#         # Use appropriate agent
#         query_lower = request.message.lower()
        
#         if any(term in query_lower for term in ['symptom', 'pain', 'fever', 'headache', 'nausea', 'ache', 'hurt']):
#             agent = symptom_analyzer_agent
#         elif any(term in query_lower for term in ['drug', 'medication', 'pill', 'dose', 'interaction', 'side effect', 'ibuprofen', 'glutathion']):
#             agent = drug_interaction_agent
#         elif any(term in query_lower for term in ['what is', 'explain', 'define', 'meaning of']):
#             agent = medical_term_agent
#         elif any(term in query_lower for term in ['report', 'result', 'test', 'lab', 'x-ray', 'summary']):
#             agent = report_analyzer_agent
#         elif any(term in query_lower for term in ['creator', 'author', 'hadiqa', 'gohar', 'medicura about', 'who made']):
#             agent = about_agent
#         else:
#             agent = general_health_agent

#         # Get intelligent response
#         result = await run_agent_with_thinking(agent, request.message, request.context)

#         # Update chat history
#         history.extend([
#             {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()},
#             {"role": "assistant", "content": result, "timestamp": datetime.now().isoformat()}
#         ])
        
#         # Keep last 20 messages
#         history = history[-20:]

#         save_history(session_id, history)

#         return result

#     except Exception as e:
#         logger.error(f"Chatbot error: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content=create_intelligent_response("I apologize for the difficulty. Please try rephrasing your question or consult a healthcare professional for immediate concerns.")
#         )

# @app.post("/api/health/drug-interactions")
# async def check_drug_interactions(input_data: DrugInteractionInput):
#     """Check drug interactions with thorough analysis"""
#     try:
#         if not input_data.medications or len(input_data.medications) == 0:
#             raise HTTPException(status_code=400, detail="At least one medication is required")
        
#         context = {
#             "medications": input_data.medications,
#             "age": input_data.age,
#             "gender": input_data.gender,
#             "existing_conditions": input_data.existing_conditions,
#             "other_medications": input_data.other_medications
#         }
        
#         prompt = f"Check interactions for: {', '.join(input_data.medications)}"
#         result = await run_agent_with_thinking(drug_interaction_agent, prompt, context)
#         return result
        
#     except Exception as e:
#         logger.error(f"Drug interaction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/medical-term")
# async def explain_medical_term(input_data: MedicalTermInput):
#     """Explain medical terms with clarity"""
#     try:
#         if not input_data.term:
#             raise HTTPException(status_code=400, detail="Medical term is required")
        
#         prompt = f"Explain the medical term: {input_data.term}"
#         if input_data.language and input_data.language != "en":
#             prompt += f" in {input_data.language} language"
        
#         result = await run_agent_with_thinking(medical_term_agent, prompt)
#         return result
        
#     except Exception as e:
#         logger.error(f"Medical term error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/health/report-summarize")
# async def summarize_medical_report(input_data: ReportTextInput):
#     """Summarize medical reports with intelligent analysis"""
#     try:
#         if not input_data.text:
#             raise HTTPException(status_code=400, detail="Report text is required")
        
#         prompt = f"""
#         Analyze and summarize this medical report:

#         {input_data.text}

#         Please provide the summary in {input_data.language if input_data.language else 'English'} language.
#         Focus on key findings, recommendations, and next steps.
#         """

#         result = await run_agent_with_thinking(report_analyzer_agent, prompt)
#         return result
        
#     except Exception as e:
#         logger.error(f"Report summary error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service temporarily unavailable")

# @app.post("/api/chatbot/session/clear")
# async def clear_session(request: ClearSessionRequest):
#     """Clear chatbot session history"""
#     try:
#         session_id = request.session_id or "default_session"
#         conn = get_db()
#         cur = conn.cursor()
#         cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
#         conn.commit()
#         cur.close()
#         conn.close()
#         return {"message": "Session cleared successfully", "session_id": session_id}
#     except Exception as e:
#         logger.error(f"Clear session error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to clear session")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "version": "2.1.0",
#         "agents_available": True,
#         "thinking_enabled": True
#     }

# @app.get("/api/chatbot/sessions")
# async def get_sessions():
#     """Get active session count (for monitoring)"""
#     conn = get_db()
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(*) FROM chat_sessions")
#     active_sessions = cur.fetchone()[0]
#     cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
#     total_messages_result = cur.fetchone()[0]
#     total_messages = total_messages_result if total_messages_result is not None else 0
#     cur.close()
#     conn.close()
#     return {
#         "active_sessions": active_sessions,
#         "total_messages": total_messages
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)








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
# import pymysql
import aiomysql  # Replace import pymysql
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
    # "database": os.getenv("TIDB_DATABASE", "test"),
    "db": os.getenv("TIDB_DATABASE", "test"),  # Changed from 'database' to 'db'
    "charset": "utf8mb4",
    # "ssl": {"ssl_mode": "VERIFY_IDENTITY"}  # Enforce SSL with hostname verification
    "ssl": True  # Simplified SSL for TiDB Cloud
}

def get_db_async():
    """Establish a connection to TiDB."""
    try:
        connection = aiomysql.connect(**DB_CONFIG)
        return connection
    # except pymysql.err.OperationalError as e:
    except aiomysql.OperationalError as e:

        logger.error(f"Failed to connect to TiDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


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

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manage application startup and shutdown."""
#     logger.info("Starting Medicura-AI Health Assistant")
#     logger.info("Connecting to TiDB...")
#     try:
#         async with (await get_db_async()) as conn:
#             async with conn.cursor() as cur:
#                 await cur.execute("""
#                     CREATE TABLE IF NOT EXISTS chat_sessions (
#                         session_id VARCHAR(100) PRIMARY KEY,
#                         history JSON NOT NULL,
#                         last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#                     )
#                 """)
#                 await cur.execute("""
#                     CREATE TABLE IF NOT EXISTS specialist_vectors (
#                         id VARCHAR(100) PRIMARY KEY,
#                         specialty VARCHAR(50) NOT NULL,
#                         content TEXT NOT NULL,
#                         embedding VECTOR(768) NOT NULL,
#                         metadata JSON,
#                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                     )
#                 """)
#                 await cur.execute("""
#                     CREATE INDEX IF NOT EXISTS idx_embedding ON specialist_vectors (embedding) 
#                     USING HNSW (metric = 'COSINE_DISTANCE', m = 16, ef_construction = 200)
#                 """)
#                 await conn.commit()
#         logger.info("TiDB connected and tables ready.")
#         yield
#     except Exception as e:
#         logger.error(f"Lifespan error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Application startup failed")
#     finally:
#         logger.info("Shutting down Medicura-AI Health Assistant")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("Starting Medicura-AI Health Assistant")
    logger.info("Connecting to TiDB...")
    try:
        async with (await get_db_async()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(100) PRIMARY KEY,
                        history JSON NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS specialist_vectors (
                        id VARCHAR(100) PRIMARY KEY,
                        specialty VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR(768) NOT NULL,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await conn.commit()
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

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
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

async def load_history(session_id: str) -> List[dict]:
    """Load chat history from TiDB."""
    try:
        async with (await get_db_async()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT history FROM chat_sessions WHERE session_id = %s", (session_id,))
                result = await cur.fetchone()
        return json.loads(result[0]) if result else []
    except Exception as e:
        logger.error(f"Failed to load history: {str(e)}")
        return []

async def save_history(session_id: str, history: List[dict]):
    """Save chat history to TiDB."""
    try:
        async with (await get_db_async()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO chat_sessions (session_id, history)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE history = %s, last_updated = CURRENT_TIMESTAMP
                """, (session_id, json.dumps(history), json.dumps(history)))
                await conn.commit()
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
@app.post("/api/chatbot/session/clear")
async def clear_session(request: ClearSessionRequest):
    """Clear chatbot session history."""
    try:
        session_id = request.session_id or "default_session"
        async with (await get_db_async()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
                await conn.commit()
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

@app.get("/api/chatbot/sessions")
async def get_sessions():
    """Get active session count (for monitoring)."""
    try:
        async with (await get_db_async()) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM chat_sessions")
                active_sessions = (await cur.fetchone())[0]
                await cur.execute("SELECT SUM(JSON_LENGTH(history)) FROM chat_sessions")
                total_messages_result = (await cur.fetchone())[0]
                total_messages = total_messages_result if total_messages_result is not None else 0
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




