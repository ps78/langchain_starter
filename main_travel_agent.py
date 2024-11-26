import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.globals import set_debug
from utils import load_certs

set_debug(True)

load_dotenv()
load_certs()

model = AzureChatOpenAI(temperature=0.2, azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))
tools = [TavilySearchResults()] 
# base for this prompt is from hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'tool_names', 'tools', 'period', 'destination', 'type'],
    template="""
    Answer the following questions as best you can. You have access to the following tools:
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

    Question: You are a professional travel agent specialized in creating individual travel experiences for your customers. You are requested to create a travel iternary for {period} to {destination}. This is a {type} trip.

    Thought:{agent_scratchpad}
    """
)
agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # type: ignore

agent_executor.invoke({
    "period": "June",
    "destination": "Thailand",
    "type": "scuba diving"
})

