from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2603"
)

loader = PyPDFLoader("./docs/PRD8 Template_v3.0 - Google Docs.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10) # 1st 1000 tokens of the 1st page of the docs

embeddings = MistralAIEmbeddings(model="mistral-embed")

chunks = text_splitter.split_documents(documents)

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="docs",
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

#3 types of Retrieval Techniques
# - Similarity Search -> Looks for similarity and can return duplicates
# - MMR (Minimum Marginal Relevance)  -> Looks for diversity and avoid returning duplicates
# - MultiQuery Retrieval -> Create multiple queries from 1 query using LLM and combine the results


retriever = vector_store.as_retriever(search_type = "mmr", search_kwargs={"k": 4, "fetch_k": 10})


template = ChatPromptTemplate.from_messages([
    ("system", '''You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document.   
 '''),
    ("human", '''
    Context: 
    {context}

    Question:
    {question}        
    ''')
])

while True:
    query = input("\nYou: ")
    if query == "exit":
        break
    context = retriever.invoke(query)
    prompt = template.format(context="\n\n".join([x.page_content for x in context]), question=query)

    response = llm.invoke(prompt)   
    print("\nAI: " + response.content)
