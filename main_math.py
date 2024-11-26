import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.agents import load_tools, AgentExecutor, create_react_agent
from langchain.globals import set_debug
from utils import load_certs

set_debug(True)
load_dotenv()
load_certs()

# prompt based on template from hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""
        Solve the following mathematical problem using basic reasoning and the following tools: {tools}
        Call the unknown 'X' and abbreviate the other parameters with distinct and suitable letters.
        Rephrase all given pieces of information as equations. 
        Then solve the system of equations using the tools you have. 
        Make sure you take in to account the time dimension. 
        Not all statements are about the present. Some are about the past and some about the future. 

        Use the following format:

        Question: the input question you must answer

        Thought: you should always think about what to do

        Action: the action to take, should be one of [{tool_names}]

        Action Input: the input to the action

        Observation: the result of the action

        ... (this Thought/Action/Action Input/Observation can repeat N times)

        Thought: I now know the final answer

        Final Answer: the final answer to the original input question

        Question: {input}
        Thought:{agent_scratchpad}
    """)

model = AzureChatOpenAI(temperature=0,azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))
tools = load_tools(["wolfram-alpha"])

agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # type: ignore

question="""
     In 10 years my sister will be 20 years older than I was 12 years ago. 
     The sum of our current ages is 16 years higher then the age of my mother.
     She was 27 when I was born. 
     How old am I?"""

answer = agent_executor.invoke({"input": question})

print(answer)