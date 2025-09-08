
# # # specialist/gastroenterology_ai.py
# # from agents import Agent, Tool

# # def create_gastroenterology_agent(model):
# #     return Agent(
# #         name="GastroenterologyAI",
# #         instructions="""
# #         You are a gastroenterology expert AI specializing in digestive system health. Analyze symptoms (e.g., abdominal pain, nausea, diarrhea, bloating) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult a gastroenterologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "gastroenterology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/gastroenterology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_gastroenterology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "gastroenterology") -> list:
#         """Search for similar gastroenterology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="GastroenterologyAI",
#         instructions="""
#         You are a gastroenterology expert AI specializing in digestive system health. Analyze symptoms (e.g., abdominal pain, nausea, diarrhea, bloating) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult a gastroenterologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "gastroenterology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar gastroenterology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_gastroenterology_agent(model):
    return Agent(
        name="MedicuraGastroenterologySpecialistAgent",
        instructions="""
You are a specialized gastroenterology agent.
You provide medical information and advice related to:
- Stomach, liver, intestines, pancreas, and digestive disorders
- Symptoms like abdominal pain, nausea, vomiting, diarrhea, constipation
- Common conditions: gastritis, ulcers, IBS, hepatitis, liver disease
- Diagnostics, treatments, lifestyle, and dietary guidance

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.
        """,
        model=model,
    )
