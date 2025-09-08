
# # # specialist/pulmonology_ai.py
# # from agents import Agent, Tool

# # def create_pulmonology_agent(model):
# #     return Agent(
# #         name="PulmonologyAI",
# #         instructions="""
# #         You are a pulmonology expert AI specializing in respiratory health assessment. Analyze respiratory symptoms (e.g., cough, wheezing, shortness of breath, chest tightness) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult a pulmonologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "pulmonology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/pulmonology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_pulmonology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "pulmonology") -> list:
#         """Search for similar pulmonology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="PulmonologyAI",
#         instructions="""
#         You are a pulmonology expert AI specializing in respiratory health assessment. Analyze respiratory symptoms (e.g., cough, wheezing, shortness of breath, chest tightness) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult a pulmonologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "pulmonology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar pulmonology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_pulmonology_agent(model):
    return Agent(
        name="MedicuraPulmonologySpecialistAgent",
        instructions="""You are a specialized pulmonology agent.
You provide medical information and advice related to:
- Respiratory conditions (asthma, COPD, bronchitis, pneumonia)
- Lung infections and tuberculosis
- Sleep apnea and breathing disorders
- Pulmonary diagnostics and treatments
- Preventive lung health tips

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
