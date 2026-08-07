from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate


loader = PyPDFLoader("./document loaders/docs/CSBS20.pdf")
pages = loader.load()

print(pages[0].page_content)