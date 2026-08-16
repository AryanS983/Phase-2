from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke on {topic}"
)

llm = ChatMistralAI(model="mistral-small-2603")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"topic":"Maths"})

print(result)

