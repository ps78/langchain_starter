import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BearlyInterpreterTool
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent

load_dotenv()

bearly_tool = BearlyInterpreterTool(api_key=os.getenv("BEARLY_API_KEY"))

tools = [bearly_tool.as_tool()]

llm = ChatOpenAI(model="gpt-4", temperature=0)

agent = initialize_agent(
    tools,
    llm,    
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

agent.run("Write python code to compute fibonacci numbers. The sequence starts like this 0, 1, 1, 2, 3,.. . The first number of the sequence is 0. Hence indexes shall be 1-based. Then run the code to generate the first 100 numbers. Return the sum of the numbers 90-100")

