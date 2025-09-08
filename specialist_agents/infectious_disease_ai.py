
# # # specialist/infectious_disease_ai.py
# # from agents import Agent, Tool

# # def create_infectious_disease_agent(model):
# #     return Agent(
# #         name="InfectiousDiseaseAI",
# #         instructions="""
# #         You are an infectious disease expert AI specializing in infection and disease analysis. Analyze symptoms (e.g., fever, sore throat, rash, fatigue) and suggest possible infections, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult an infectious disease specialist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "infectious_disease"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/infectious_disease_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_infectious_disease_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "infectious_disease") -> list:
#         """Search for similar infectious disease cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="InfectiousDiseaseAI",
#         instructions="""
#         You are an infectious disease expert AI specializing in infection and disease analysis. Analyze symptoms (e.g., fever, sore throat, rash, fatigue) and suggest possible infections, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult an infectious disease specialist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "infectious_disease"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar infectious disease cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_infectious_disease_agent(model):
    return Agent(
        name="MedicuraInfectiousDiseaseSpecialistAgent",
        instructions="""
You are a specialized infectious disease agent.
You provide medical insights related to:
- Viral, bacterial, fungal, and parasitic infections
- Symptoms, transmission, prevention, and treatment
- Diagnostic recommendations and common complications
- Guidance on isolation, vaccination, and public health precautions

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.
        """,
        model=model,
    )
