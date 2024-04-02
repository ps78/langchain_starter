import os
import getpass
import platform
import requests
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import BearlyInterpreterTool
from langchain.agents import AgentType, create_openai_tools_agent, AgentExecutor, create_openai_functions_agent
from langchain.globals import set_debug
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from utils import load_certs

set_debug(True)
load_dotenv()
load_certs()

bearly_tool = BearlyInterpreterTool(api_key=os.getenv("BEARLY_API_KEY")).as_tool()

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [repl_tool]

llm = AzureChatOpenAI(temperature=0.0, azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))

prompt = hub.pull("chuxij/open-interpreter-system")
prompt.input_variables.append("agent_scratchpad")
username = getpass.getuser()
current_working_directory = os.getcwd()
operating_system = platform.system()
info = f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"

query = "write python code to check if a number is prime or not"
url = "https://open-procedures.replit.app/search/"
relevant_procedures = requests.get(url, params={"query": query}).json()["procedures"]
info += "\n\n# Recommended Procedures\n" + "\n---\n".join(relevant_procedures) + "\nIn your plan, include steps and, if present, **EXACT CODE SNIPPETS** (especially for depracation notices, **WRITE THEM INTO YOUR PLAN -- underneath each numbered step** as they will VANISH once you execute your first line of code, so WRITE THEM DOWN NOW if you need them) from the above procedures if they are relevant to the task. Again, include **VERBATIM CODE SNIPPETS** from the procedures above if they are relevent to the task **directly in your plan.**"

system_message = "\n\n" + info

function_schema = {
  "name": "run_code",
  "description":
  "Executes code on the user's machine and returns the output",
  "parameters": {
    "type": "object",
    "properties": {
      "language": {
        "type": "string",
        "description":
        "The programming language",
        "enum": ["python", "shell", "applescript", "javascript", "html"]
      },
      "code": {
        "type": "string",
        "description": "The code to execute"
      }
    },
    "required": ["language", "code"]
  },
}

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # type: ignore

print(prompt)

result = agent_executor.invoke({
    "question": query,
    "current_working_directory": current_working_directory,
    "operating_system": operating_system,
    "relevant_procedures": relevant_procedures,
    "username": username
})

print(result)