
# # # specialist/dental_ai.py
# # from agents import Agent, Tool

# # def create_dental_agent(model):
# #     return Agent(
# #         name="DentalAI",
# #         instructions="""
# #         You are a dental expert AI specializing in oral health assessment. Analyze oral symptoms (e.g., toothache, gum swelling, bad breath, tooth sensitivity) and suggest possible conditions, treatments, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, treatments, or specialist referrals (e.g., consult a dentist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "dental"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )

# # specialist/dental_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_dental_agent(model):
#     from main import search_similar_cases

#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "dental") -> list:
#         """Search for similar dental cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="DentalAI",
#         instructions="""
#         You are a dental expert AI specializing in oral health assessment. Analyze oral symptoms (e.g., toothache, gum swelling, bad breath, tooth sensitivity) and suggest possible conditions, treatments, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, treatments, or specialist referrals (e.g., consult a dentist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "dental"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar dental cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )


from agents import Agent

def create_dental_agent(model):
    return Agent(
        name="MedicuraDentalSpecialistAgent",
        instructions="""You are a specialized dental agent.
You provide medical information and advice related to:
- Oral health and hygiene
- Tooth decay, cavities, and gum diseases
- Orthodontics (braces, alignment issues)
- Dental procedures (fillings, root canals, implants)
- Preventive dental care and lifestyle recommendations

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
