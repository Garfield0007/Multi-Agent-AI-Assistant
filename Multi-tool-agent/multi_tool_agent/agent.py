from google.adk.agents import Agent
from .rag_system import search_knowledge_base

#---------------Tool 1: hello tool-----------------
def say_hello(name: str):
    return {
        "status": "success",
        "report": f"Hello {name}, your ADK multi-agent is working!"
    }
#-----------------greeting agent------------------
greeting_agent = Agent(
    name="greeting_agent",
    model="gemini-2.5-flash",
    description="A simple agent that greets the user.",
    instruction="Use say_hello tool when the user greets you.",
    tools=[say_hello]
)
#-----------------Calculator tool-----------------
def calculator(expression: str):
    try:
        answer = eval(expression)
        return {
            "status": "success",
            "report": f"The answer is {answer}."
        }
    
    except Exception as error:
        return{
            "status": "error",
            "error_message": str(error)
        }

#-----------------Math Agent-----------------
math_agent =Agent(
    name="math_agent",
    model="gemini-2.5-flash",
    description="handles mathematical calculations",
    instruction="Use the calculator tool when the user asks you to perform a calculation.",
    tools=[calculator]
)


#-----------------RAG TOOL-----------------
def rag_search(question: str):
    try:
        result = search_knowledge_base(question)
        return {
            "status": "success",
            "report": result
        }
    except Exception as error:
        return {
            "status": "error",
            "error_message": str(error)
        }

#-----------------RAG Agent-----------------
rag_agent = Agent(
    name="rag_agent",
    model="gemini-2.5-flash",
    description="handles retrieval-augmented generation queries",
    instruction="Use the rag_search tool when the user asks you a question that requires information retrieval.",
    tools=[rag_search]
)

#-----------------Root Router Agent-----------------
root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="Main routing agent.",
    instruction="You are the root routing agent. Delegate:"
    "-math questions to math_agent,"
    "-greeting to greeting_agent,",
    sub_agents=[
    greeting_agent,
    math_agent,
    rag_agent
    ]
)