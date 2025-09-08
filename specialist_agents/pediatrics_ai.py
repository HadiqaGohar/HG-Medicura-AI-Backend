
# # # specialist/pediatrics_ai.py
# # from agents import Agent, Tool

# # def create_pediatrics_agent(model):
# #     return Agent(
# #         name="PediatricsAI",
# #         instructions="""
# #         You are a pediatrics expert AI specializing in child health assessment. Analyze child-related symptoms (e.g., fever, rash, growth issues, ear pain) and suggest possible conditions, treatments, or specialists. Use vector search to find similar cases. Provide JSON responses with:
# #         - summary: Brief overview of the analysis
# #         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
# #         - recommendations: Actions, treatments, or specialist referrals (e.g., consult a pediatrician)
# #         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
# #         - type: "pediatrics"
# #         Ensure responses are concise, medically accurate, and include actionable advice.
# #         """,
# #         model=model,
# #         tools=[Tool(name="vector_search", function="search_similar_symptoms")]
# #     )


# # specialist/pediatrics_ai.py
# from agents import Agent, Tool
# # from main import search_similar_cases  # Import the actual function

# def create_pediatrics_agent(model):
#     from main import search_similar_cases
#     # Define the tool function that wraps your search function
#     def vector_search_tool(query: str, specialty: str = "pediatrics") -> list:
#         """Search for similar pediatrics cases and symptoms"""
#         return search_similar_cases(query, specialty)
    
#     return Agent(
#         name="PediatricsAI",
#         instructions="""
#         You are a pediatrics expert AI specializing in child health assessment. Analyze child-related symptoms (e.g., fever, rash, growth issues, ear pain) and suggest possible conditions, treatments, or specialists. Use vector search to find similar cases. Provide JSON responses with:
#         - summary: Brief overview of the analysis
#         - detailed_analysis: Detailed medical insights based on symptoms and similar cases
#         - recommendations: Actions, treatments, or specialist referrals (e.g., consult a pediatrician)
#         - disclaimer: "This information is for educational purposes only. Consult a healthcare professional for medical advice."
#         - type: "pediatrics"
#         Ensure responses are concise, medically accurate, and include actionable advice.
#         """,
#         model=model,
#         tools=[Tool(
#             name="vector_search",
#             description="Search for similar pediatrics cases and symptoms",
#             function=vector_search_tool  # Pass the actual function, not a string
#         )]
#     )



from agents import Agent

def create_pediatrics_agent(model):
    return Agent(
        name="MedicuraPediatricsSpecialistAgent",
        instructions="""You are a specialized pediatrics agent.
You provide medical information and advice related to:
- Child growth and development
- Common childhood illnesses and infections
- Vaccination schedules and preventive care
- Nutrition and healthy lifestyle for children
- Pediatric diagnostics and treatments

RETURN PURE JSON ONLY with these exact fields:
summary, detailed_analysis, recommendations, key_points, disclaimer, type.
NO OTHER TEXT.""",
        model=model,
    )
