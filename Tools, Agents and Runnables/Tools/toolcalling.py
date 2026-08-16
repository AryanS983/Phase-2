from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from rich import print


#creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in the input text."""
    return len(text)


#using the tool
llm = ChatMistralAI(model="mistral-small-2603")

#tool binding 
llm_with_tool = llm.bind_tools([get_text_length])


result = llm_with_tool.invoke("Return the number of characters in the given text: 'How are you ?' ")

if result.tool_calls:
    tool_call = result.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_result = get_text_length.invoke(tool_args)
    final_result = llm_with_tool.invoke(f"The number of characters in the given text is: {tool_result}")

    print(final_result)
