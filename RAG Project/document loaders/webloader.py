from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2603"
)

loader = WebBaseLoader("https://in.bookmyshow.com/movies/kolkata/spider-man-brand-new-day/ET00447840")

documents = loader.load()

template = ChatPromptTemplate.from_messages([
    ("system", '''You are a web scraper tool.
    Extract the data about the movie from the page.    
 '''),
    ("human", "{data}")
])

prompt = template.format(data=documents[0].page_content)

response = llm.invoke(prompt)

print(response.content)
