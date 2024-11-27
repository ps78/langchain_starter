#
# Answers question about a youtube video
#
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

load_dotenv()

embeddings = AzureOpenAIEmbeddings(azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"))

def create_vector_db_from_youtube_url(video_url :str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = AzureChatOpenAI(azure_deployment=os.getenv("AZURE_GPT4_DEPLOYMENT"))
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You are a helpful agent who can answer about videos based on their transcript.
            Search the following transcripts {docs}.

            Answer the following question {question}.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

#
# Main
#
db = create_vector_db_from_youtube_url("https://www.youtube.com/watch?v=U9mJuUkhUzk")
answer= get_response_from_query(db, "Why does Sam Altman not like pink elephants?", 5)
print(answer)