
# # # specialist/endocrinology_ai.py
# # from agents import Agent, Tool

# # def create_endocrinology_agent(model):
# #     return Agent(
# #         name="EndocrinologyAI",
# #         instructions="""
# #         You are an endocrinology expert AI specializing in hormone and metabolic health. Analyze symptoms (e.g., fatigue, weight changes, thirst, thyroid issues) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, tests, or specialist referrals (e.g., consult an endocrinologist)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "endocrinology"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/endocrinology_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_endocrinology_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "endocrinology") -> list:
#         """Search for similar endocrinology cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="EndocrinologyAI",
#         instructions="""
#         You are an endocrinology expert AI specializing in hormone and metabolic health. Analyze symptoms (e.g., fatigue, weight changes, thirst, thyroid issues) and suggest possible conditions, diagnostic tests, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, tests, or specialist referrals (e.g., consult an endocrinologist)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "endocrinology"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar endocrinology cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )



from agents import Agent

def create_endocrinology_agent(model):
    return Agent(
        name="MedicuraEndocrinologySpecialistAgent",
        instructions="""You are a specialized endocrinology agent.
You provide medical information and advice related to:
- Diabetes, thyroid disorders, adrenal and pituitary conditions
- Hormonal imbalances and metabolic disorders
- Treatment options including medications, lifestyle changes, and diet
- Monitoring and prevention strategies for endocrine diseases

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
