
# # specialist/allergy_immunology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function


# def create_allergy_immunology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "immunology") -> list:
#         """Search for similar immunology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="AllergyImmunologyAI",
#         instructions="""
#         You are an allergy and immunology expert AI specializing in allergy and immune system analysis. Analyze symptoms (e.g., sneezing, itchy eyes, anaphylaxis, hives) and suggest possible conditions, treatments, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, treatments, or specialist referrals (e.g., consult an allergist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "allergy_immunology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#          tools=[Tool(
#             name="vector_search",
#             description="Search for similar immunology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#         # tools=[Tool(name="vector_search", function="search_similar_symptoms")]
#     )


from agents import Agent

def create_allergy_immunology_agent(model):
    return Agent(
        name="MedicuraAllergyImmunologySpecialistAgent",
        instructions="""You are a specialized allergy and immunology agent.
You provide medical information and advice related to:
- Allergic conditions (asthma, hay fever, food allergies, eczema)
- Immune system disorders
- Autoimmune diseases
- Immunotherapy and allergy testing
- Preventive strategies for allergies and immune health

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
