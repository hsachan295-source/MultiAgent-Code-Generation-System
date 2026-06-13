from dotenv import load_dotenv
load_dotenv()
import os #ishka use .env veriable hai usko access ker sake

from langchain_mistralai.chat_models  import ChatMistralAI
from langchain.agents import create_agent
import subprocess # subprocess module use hota hai Python ke andar external programs,
import sys # sys module Python interpreter se related information aur
from langchain_core.tools import tool
from langchain.messages import HumanMessage,AIMessage

 
  
model = ChatMistralAI(
  model="mistral-medium-latest",
  api_key=os.getenv("MISTRAL_API_KEY")
)

coder_agent = create_agent(model=model,
                            tools=[],
                            system_prompt="""
                            You are an experienced software developer.
                            You write Python code with proper understanding of the problem statement.
                            You write code with proper comments and docstrings.
                            You think about edge cases and error handling while writing code.
                            """)

planner_agent = create_agent(model=model,
                             tools=[],
                             system_prompt="""You are an experienced Software Architect.
                             your task is to understand the problem statement and create a plan to solve the problem
                               
""")

 
@tool
def execute_code(code: str) -> str:
  """
  Use this tool to execute the python code , find out the code is working or not
  
  Args:
       code (str): python code to execute
       
   """
  print("🔥 Execute Code Tool Called")
  
  result = subprocess.run(
    [sys.executable,"-c",code], #path interpreater
    capture_output = True,#stdout (normal output) aur stderr (error output),# dono ko capture karke result object me store karega
    text = True, # Output bytes me nahi balki string format me milega
    timeout=30  # Agar code 30 seconds se zyada chale, to process automatically stop ho jayega

  )

  return str({
    "stdout": result.stdout, #Example: print("Hello") --> Hello
    "stderr": result.stderr,  #Error messages,# Example: SyntaxError, NameError
    "returncode": result.returncode # Program successful hua ya nahi--->0  = Success ,Non-zero = Error

  })
 
tester_agent = create_agent(

  model=model,
  tools=[execute_code],
  system_prompt = """
  You are an experienced software tester.
  your task is to test the python code and find out the code is working or not.
  You can use the tool to execute the code and find out the result."""
)

@tool
def planner_tool(problem_statement:str)->str:
  """
  Use this tool to create a plan to solve the problem statement
  
  arg:
     problem_statement (str): problem statement to create a  plan
  """
  print("Planner tool called with problem statement:",problem_statement)

  response = planner_agent.invoke({
    "messages":[
      HumanMessage(problem_statement)
    ]
  })
  response['messages'][-1].pretty_print()
  return response["messages"][-1].content

@tool
def coder_tool(plan:str)->str:
  """
  Use this tool to write python code to solve the problem statement
  
  Args:
      Plan (str): Plan to write python code"""
  
  print("coder tool called with plan:" ,plan)
  
  response = coder_agent.invoke({
    "messages": [
      HumanMessage(plan)
    ]
  })
  response['messages'][-1].pretty_print()

  return response["messages"][-1].content

 
@tool
def tester_tool(code: str) -> str:
    """
    Use this tool to test the python code and find out
    whether the code is working or not.

    Args:
        code (str): Python code to test
    """
    print("Tester tool called with code:",code)
    response = tester_agent.invoke(
        {
            "messages": [
                HumanMessage(code)
            ]
        }
    )
    response['messages'][-1].pretty_print()

    return response["messages"][-1].content


team_lead_agent = create_agent(
    model=model,
    tools=[planner_tool, coder_tool, tester_tool],
    system_prompt="""
    You are an experienced software team lead.

    Your task is to understand the problem statement and create a plan
    to solve the problem.

    Then you need to write Python code to solve the problem statement.

    After that, you need to test the Python code and determine
    whether the code is working correctly or not.

    You can use the available tools to create a plan,
    write code, and test the code.
    """
)


response = team_lead_agent.invoke({
   "messages":[
      HumanMessage("Write a python function to find the factorial of number")
   ]
})

print(response["messages"][-1].pretty_print())