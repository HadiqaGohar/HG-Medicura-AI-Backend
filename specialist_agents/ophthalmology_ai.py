
# # # specialist/ophthalmology_ai.py
# # from agents import Agent, Tool

# # def create_ophthalmology_agent(model):
# #     return Agent(
# #         name="OphthalmologyAI",
# #         instructions="""
# #         You are an ophthalmology expert AI specializing in eye health analysis. Analyze eye-related symptoms (e.g., blurry vision, eye pain, redness, floaters) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult an ophthalmologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "ophthalmology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )



# # specialist/ophthalmology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_ophthalmology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "ophthalmology") -> list:
#         """Search for similar ophthalmology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="OphthalmologyAI",
#         instructions="""
#         You are an ophthalmology expert AI specializing in eye health analysis. Analyze eye-related symptoms (e.g., blurry vision, eye pain, redness, floaters) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult an ophthalmologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "ophthalmology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar ophthalmology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_ophthalmology_agent(model):
    return Agent(
        name="MedicuraOphthalmologySpecialistAgent",
        instructions="""You are a specialized ophthalmology agent.
You provide medical information and advice related to:
- Eye diseases (glaucoma, cataracts, conjunctivitis, macular degeneration)
- Vision problems (myopia, hyperopia, astigmatism, presbyopia)
- Eye trauma and infections
- Preventive eye care and lifestyle recommendations
- Diagnostics and treatment options in ophthalmology

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
