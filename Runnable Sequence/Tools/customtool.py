from langchain.tools import tool

@tool   #Decorator for creating custom tool
def custom_tool(name: str) -> str:
    """Return a greeting message for the provided `name`.
    """     #Docstring - Used to describe the tool and is required in a tool as it is used by llms to understand what the tool does
    return f"Hello your input was: {name}"

get_greeting = custom_tool.invoke({"name": "Johny"})

print(get_greeting)
print(custom_tool.args) #attribute of the tool

