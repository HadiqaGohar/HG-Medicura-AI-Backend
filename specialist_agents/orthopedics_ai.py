
# # # specialist/orthopedics_ai.py
# # from agents import Agent, Tool

# # def create_orthopedics_agent(model):
# #     return Agent(
# #         name="OrthopedicsAI",
# #         instructions="""
# #         You are an orthopedics expert AI specializing in bone and joint health. Analyze symptoms (e.g., joint pain, swelling, fracture, stiffness) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult an orthopedist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "orthopedics"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )

# # specialist/orthopedics_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_orthopedics_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "orthopedics") -> list:
#         """Search for similar orthopedics cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="OrthopedicsAI",
#         instructions="""
#         You are an orthopedics expert AI specializing in bone and joint health. Analyze symptoms (e.g., joint pain, swelling, fracture, stiffness) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult an orthopedist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "orthopedics"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar orthopedics cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )



from agents import Agent

def create_orthopedics_agent(model):
    return Agent(
        name="MedicuraOrthopedicsSpecialistAgent",
        instructions="""You are a specialized orthopedics agent.
You provide medical information and advice related to:
- Bone fractures, injuries, and trauma
- Joint problems (arthritis, dislocation, replacement)
- Spine disorders and posture issues
- Sports medicine and rehabilitation
- Preventive care for bone and joint health

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
