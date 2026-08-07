from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2603"
)

loader = PyPDFLoader("./document loaders/docs/CSBS20.pdf")

documents = loader.load()

template = ChatPromptTemplate.from_messages([
    ("system", '''You are an expert data extraction tool.

Extract the data into CSV format.

Rules:
1. Do NOT wrap the CSV in ``` or markdown.
2. First line MUST be the header.
3. Every record MUST be on a new line.
4. Separate columns with commas.
5. If a value is missing, leave it blank.
       
Example:
Sl
No.
Category Subject Code Subject Name
L 
T
P
Credits        
 '''),
    ("human", "{data}")
])

prompt = template.format(data=list(map(lambda x: x.page_content, documents)))

response = llm.invoke(prompt)

with open("document loaders/docs/new1.csv", "w", encoding="utf-8") as f:
    f.write(response.content)

print(response.content)
