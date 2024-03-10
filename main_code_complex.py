import os
from uuid import uuid4 # type: ignore
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import load_tools, AgentType, AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain import hub
from langchain.globals import set_debug
set_debug(True)

load_dotenv()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"

# You can create the tool to pass to an agent
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

model = AzureChatOpenAI(temperature=0.2, azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))
tools = [repl_tool]
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'tool_names', 'tools', 'task'],
    template="""
    Write Python code that solves the following task:
    {task}
    
    You have access to these tools: 
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer\nFinal Answer: the final answer to the original input question
    
    Begin!
    
    Thought:{agent_scratchpad}
    """
)
agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # type: ignore

agent_executor.invoke({
    "task": "Create a function that returns the n-th Fibonacci number, where n is a parameter of the function.",
})
