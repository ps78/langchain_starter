import os
from dotenv import load_dotenv
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_community.tools import BearlyInterpreterTool
from langchain_openai import AzureChatOpenAI
from langchain_experimental.utilities import PythonREPL

load_dotenv()

search = DuckDuckGoSearchAPIWrapper()
llm = AzureChatOpenAI(temperature=0, azure_deployment=os.getenv("AZURE_GPT3_DEPLOYMENT"))
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. The input argument must be a single string",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    BearlyInterpreterTool(api_key=os.getenv("BEARLY_API_KEY")).as_tool(),
    Tool(
        name="python_repl",
        func=python_repl.run,
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
    )
]

model = AzureChatOpenAI(temperature=0, azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True) # type: ignore
agent = PlanAndExecute(planner=planner, executor=executor)

agent.invoke({"input": "Write a Python program that simulates the solar systems, consisting of the sun and the 8 planets. It shall use the known masses and distances of these celestial objects and apply Newtons gravitation law. Start the simulation with all planets and the sun aligned on a line and positioned at their respective known distances from the sun. Then run the simulation for a period of 1000 years."})