import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug
from utils import load_certs

set_debug(True)
load_dotenv()
load_certs()

def generate_chain():
    prompt_template = PromptTemplate(
        input_variables=["ingredients", "course"],
        template="You are a professional chef and expert in cooking all kinds of foods from around the world. You are also very proficient in creating new recipes. Given the following list of ingredients, create a recipe for an inspiring {course}. These are the ingredients:\n {ingredients}\n\nUse metric units in your recipe descriptions, like °C, ml, g. Do not use °F, cups, ounces, pounds."
    )
    model = AzureChatOpenAI(temperature=0.8, azure_deployment=os.getenv("AZURE_GPT3_DEPLOYMENT"))

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