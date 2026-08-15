from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from rich import print


#creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in the input text."""
    return len(text)


#using the tool
llm = ChatMistralAI(model="mistral-small-2603")

#tool binding
llm.bind_tools