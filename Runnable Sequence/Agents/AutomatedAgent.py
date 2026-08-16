from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

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

@wrap_tool_call
def human_approval(request, handler):
    '''Ask for human approval before executing the tool'''
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent: Do you want to execute {tool_name}? (yes/no): ")
    if confirm.lower() == "yes":
        return handler(request)
    else:
        return ToolMessage(content="Tool execution cancelled by user.", tool_call_id=request.tool_call["id"])



llm = ChatMistralAI(model="mistral-small-2603")

agent = create_agent(
    llm, 
    tools=[get_weather, get_news], 
    system_prompt="You are a helpful city assistant",
    middleware=[human_approval]
)

print("city agent")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    human_message = {"messages": [{"role": "user", "content": user_input}]}
    result = agent.invoke(human_message)
    print(result["messages"][-1].content)

