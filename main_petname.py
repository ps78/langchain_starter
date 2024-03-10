#
# Generates pet names, based od: https://www.youtube.com/watch?v=lG7Uxts9SXs
#
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()

def generate_pet_name(animal_type :str, color :str, n :int) -> str:
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "color", "n"],
        template="I have a {color} {animal_type} pet and I want a cool name for it. Suggest me {n} cool names for my pet"
    )
    llm = AzureChatOpenAI(temperature=0.8, azure_deployment=os.getenv("AZURE_GPT3_DEPLOYMENT"))
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain({
        "animal_type": animal_type,
        "color": color,
         "n": n
    })
    return response["pet_name"]

def langchain_agent():
    llm = AzureChatOpenAI(temperature=0.5, azure_deployment=os.getenv("AZURE_GPT3_DEPLOYMENT"))
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    result = agent.run("What is the average age of a dog? Multiply the age by 3")
    print(result)

#
# Main
#
print(generate_pet_name("dog", "pink", 5))
langchain_agent()