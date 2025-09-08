# # from agents import Agent, Tool

# # def create_cardiology_agent(model):
# #     return Agent(
# #         name="CardiologyAI",
# #         instructions="""
# #         You are a cardiology expert AI specializing in heart health assessment. Analyze heart-related symptoms (e.g., chest pain, palpitations, shortness of breath, fatigue) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases from the database. Provide JSON responses with the following structure:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult a cardiologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "cardiology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )

# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function


# def create_cardiology_agent(model):
#     from main import search_similar_cases

#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "cardiology") -> list:
#         """Search for similar cardiology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="CardiologyAI",
#         instructions="""
#         You are a cardiology expert AI specializing in heart health assessment. Analyze heart-related symptoms (e.g., chest pain, palpitations, shortness of breath, fatigue) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases from the database. Provide JSON responses with the following structure:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult a cardiologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "cardiology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar cardiology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_cardiology_agent(model):
    return Agent(
        name="MedicuraCardiologySpecialistAgent",
        instructions="""You are a specialized cardiology agent.
You provide medical information and advice related to:
- Heart diseases
- Hypertension
- Cholesterol
- Cardiac diagnostics and treatments
- Preventive cardiology tips

RETURN PURE JSON ONLY with these exact fields: 
summary, detailed_analysis, recommendations, key_points, disclaimer, type. 
NO OTHER TEXT.""",
        model=model,
    )
