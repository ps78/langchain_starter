from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
import langchain 

langchain.debug = True

load_dotenv()

def generate_chain() -> LLMChain:
    prompt_template = PromptTemplate(
        input_variables=["ingredients", "course"],
        template="You are a professional chef and expert in cooking all kinds of foods from around the world. You are also very proficient in creating new recipes. Given the following list of ingredients, create a recipe for an inspiring {course}. These are the ingredients:\n {ingredients}\n\nUse metric units in your recipe descriptions, like °C, ml, g. Do not use °F, cups, ounces, pounds."
    )   
    model = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo-instruct")
    
    chain = (
        RunnablePassthrough()
        | prompt_template
        | model
        | StrOutputParser()
    )

    return chain

chain = generate_chain()
result = chain.invoke(input={
    "ingredients": [
        "bacon", "chorizo", 
        "potatoes", "tomatoes", "egg plants", "fennel", 
        "cream", "parmesan cheese"
    ], 
    "course": "starter"})
print(result)