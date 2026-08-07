from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "mistralai:mistral-small-2603"
)

response = model.invoke("3 line paragraph on Mutual Funds")

print(response.content)