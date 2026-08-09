from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma


load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

docs = [
    Document(page_content="This is a Python document. Its used for AIML", metadata={"extension": "py"}),
    Document(page_content="This is Java document. It is used for application development", metadata={"extension": "java"}),
    Document(page_content="This is C++ document. It is used for making games", metadata={"extension": "cpp"}),
    Document(page_content="This is C document. It is used for making OS", metadata={"extension": "c"}),
]

ids = [f"doc_{i}" for i in range(len(docs))]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=ids,
    collection_name="programming_languages",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


result =  vector_store.similarity_search("What is used for making games", k=1)

print(result)
