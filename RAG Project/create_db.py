#load pdf
#split into chunks
#create embeddings
#store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()


loader = PyPDFLoader("./docs/PRD8 Template_v3.0 - Google Docs.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10) # 1st 1000 tokens of the 1st page of the docs

chunks = text_splitter.split_documents(documents)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="docs",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)



# retriever =  vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# result = retriever.invoke("Which ML Technique is used in the project?")
# print(result)