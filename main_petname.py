import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from utils import load_certs

load_certs()
load_dotenv()

def generate_pet_name(animal_type :str, color :str, n :int) -> str:
    prompt = PromptTemplate(
        input_variables=["animal_type", "color", "n"],
        template="I have a {color} {animal_type} pet and I want a cool name for it. Suggest me {n} names for my pet"
    )

    model = AzureChatOpenAI(temperature=0.8, azure_deployment=os.getenv("AZURE_GPT3_DEPLOYMENT"))

    name_chain = (
        RunnablePassthrough()
        | prompt
        | model
        | StrOutputParser()
    )

    response = name_chain.invoke({
        "animal_type": animal_type,
        "color": color,
         "n": n
    })
    return response

print(generate_pet_name("snake", "rainbow", 10))