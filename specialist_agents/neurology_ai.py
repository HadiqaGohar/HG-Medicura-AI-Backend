# # # specialist/neurology_ai.py
# # from agents import Agent, Tool

# # def create_neurology_agent(model):
# #     return Agent(
# #         name="NeurologyAI",
# #         instructions="""
# #         You are a neurology expert AI specializing in neurological symptom analysis. Analyze symptoms (e.g., headache, dizziness, numbness, seizures) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult a neurologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "neurology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/neurology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_neurology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "neurology") -> list:
#         """Search for similar neurology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="NeurologyAI",
#         instructions="""
#         You are a neurology expert AI specializing in neurological symptom analysis. Analyze symptoms (e.g., headache, dizziness, numbness, seizures) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult a neurologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "neurology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar neurology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )

from agents import Agent

def create_neurology_agent(model):
    return Agent(
        name="MedicuraNeurologySpecialistAgent",
        instructions="""You are a specialized neurology agent.
You provide medical information and advice related to:
- Brain disorders (stroke, epilepsy, migraine, multiple sclerosis)
- Nerve and spinal cord conditions
- Neurodegenerative diseases (Alzheimer’s, Parkinson’s)
- Sleep disorders
- Preventive neurological health tips

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
