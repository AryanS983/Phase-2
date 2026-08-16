from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None,
    # country=None
    # include_favicon=False
    # include_usage=False
)

llm = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_template('''Summarize the following news into bullet points: 
{news}
''')

output_parser = StrOutputParser()



news = tool.run("Latest AI news of 2026")

chain =  prompt | llm | output_parser

print(chain.invoke({"news":news}))
