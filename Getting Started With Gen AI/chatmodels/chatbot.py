from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


model = init_chat_model(
    "mistralai:mistral-small-2603"
)

messages = [
    SystemMessage(content="You are an experienced Computer Science Teacher")
]


print("Hello Im your chatbot")

while True:
    user_input = input("You: ")
    if user_input == "0":
        break
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    print("Chatbot:", response.content)
    messages.append(AIMessage(content=response.content))

print(messages)