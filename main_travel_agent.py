from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
import langchain 

langchain.debug = True

load_dotenv()

def generate_chain(model) -> LLMChain:
    prompt_template = PromptTemplate(
        input_variables=["period", "destination", "type"],
        template="You are an professional travel agent specialized in creating individual tranvel experiences for your customers. You are requested to create a travel iternary for {period} to {destination}. This is a {type} trip."
    )
    
    chain = (
        RunnablePassthrough()
        | prompt_template
        | model
        | StrOutputParser()
    )

    return chain

model = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct")
web_search = DuckDuckGoSearchRun()
agent = initialize_agent(llm=model, tools=[web_search], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

chain = generate_chain(model)
result = chain.invoke(input={
    "period": "next summer", 
    "destination": "Thailand",
    "type": "family"})
print(result)

#search = GoogleSearchAPIWrapper()
#tool = Tool(
#    name="Google Search",
#    description="Search Google for recent results.",
#    func=search.run,
#)
