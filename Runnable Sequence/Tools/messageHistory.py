from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

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

message = []
query = HumanMessage(content="Return the number of characters in the given text: 'How are you ?' ")
message.append(query)

result = llm_with_tool.invoke(message)
message.append(result)

tool_mapping = {
    "get_text_length": get_text_length
}

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tool_mapping[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

final_result = llm_with_tool.invoke(message)
print(message)
print(final_result.content)


