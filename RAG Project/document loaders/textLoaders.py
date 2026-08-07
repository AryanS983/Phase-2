from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2603"
)

loader = TextLoader("./document loaders/docs/emp.txt")

documents = loader.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a tool which creates a csv from data."),
    ("human", "{data}")
])

prompt = template.format(data=documents)

response = llm.invoke(prompt)

print(response.content)
