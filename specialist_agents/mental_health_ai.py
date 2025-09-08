
# # # specialist/mental_health_ai.py
# # from agents import Agent, Tool

# # def create_mental_health_agent(model):
# #     return Agent(
# #         name="MentalHealthAI",
# #         instructions="""
# #         You are a mental health expert AI specializing in mental wellness assessment. Analyze symptoms (e.g., anxiety, depression, insomnia, stress) and suggest possible conditions, coping strategies, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed insights based on symptoms and similar cases
# #         - recommendations: Actions, strategies, or specialist referrals (e.g., consult a psychiatrist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "mental_health"
# #         Ensure responses are concise, empathetic, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )

# # specialist/mental_health_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_mental_health_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "mental_health") -> list:
#         """Search for similar mental health cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="MentalHealthAI",
#         instructions="""
#         You are a mental health expert AI specializing in mental wellness assessment. Analyze symptoms (e.g., anxiety, depression, insomnia, stress) and suggest possible conditions, coping strategies, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed insights based on symptoms and similar cases
#         - recommendations: Actions, strategies, or specialist referrals (e.g., consult a psychiatrist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "mental_health"
#         Ensure responses are concise, empathetic, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar mental health cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )



from agents import Agent

def create_mental_health_agent(model):
    return Agent(
        name="MedicuraMentalHealthSpecialistAgent",
        instructions="""You are a specialized mental health agent.
You provide medical information and advice related to:
- Depression, anxiety, and stress management
- Bipolar disorder, schizophrenia, and other psychiatric conditions
- Counseling techniques and therapy options
- Lifestyle recommendations for mental well-being
- Coping strategies, mindfulness, and relaxation techniques

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
