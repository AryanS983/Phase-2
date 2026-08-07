from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2603"
)

loader = PyPDFLoader("./document loaders/docs/CSBS20.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10) # 1st 1000 tokens of the 1st page of the docs

chunks = text_splitter.split_documents(documents)

template = ChatPromptTemplate.from_messages([
    ("system", '''You are an expert data Summarization tool.  
    Summerize the following information.   
 '''),
    ("human", "{data}")
])

prompt = template.format(data=list(map(lambda x: x.page_content, documents)))

response = llm.invoke(prompt)

print(response.content)
