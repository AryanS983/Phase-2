from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

from rich import print
import os
import requests

#Weater Tool

@tool
def get_weather(city: str)-> str:
    '''Get Current Weather of a City'''
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return {
            "error": "Unable to fetch weather"
        }

    data = response.json()

    return f"The current weather in {city} is {data['weather'][0]['description']} with a temperature of {round(data['main']['temp'] - 273.15)} degrees Celsius."


#Tavily News Tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def get_news(city: str)-> str:
    '''Get the latest news about the city'''
    response = tavily_client.search(
        query=f"latest news about {city}",
        city="news",
        search_depth="basic",
        max_results=5
    )

    results = response.get("results", [])

    if not results:
        return f"No recent news found for {city}."

    news = []

    for result in results:
        title = result.get("title", "No title")
        content = result.get("content", "No description available")
        url = result.get("url", "")

        news.append(
            f"Title: {title}\n"
            f"Summary: {content[:100]}\n"
            f"Source: {url}"
        )

    return "\n\n".join(news)


llm = ChatMistralAI(model="mistral-small-2603")
llm_with_tool = llm.bind_tools([get_weather, get_news])

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

#Agent Loop

messages = []

while True:
    user_input = input("User: ")
    human_message = HumanMessage(content=user_input)
    messages.append(human_message)

    if user_input.lower() == "exit":
        break

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        #if tool required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]

                confirm = input(f"Tool {tool_name} required. Do you want to use it? (y/n) ")
                if confirm.lower() != "y":
                    continue

                #execute tool
                tool_result = tools[tool_name].invoke(tool_call)
                messages.append(tool_result)
        else:
            print(result.content)
            break
            
