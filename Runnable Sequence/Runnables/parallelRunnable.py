from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda


short = ChatPromptTemplate.from_template(
    "tell me a short joke on {topic}"
)

detailed = ChatPromptTemplate.from_template(
    "tell me a detailed joke on {topic}"
)

llm = ChatMistralAI(model="mistral-small-2603")

output_parser = StrOutputParser()

#For parallel execution of runnables we use dictionary. Since dictionary is not a runnable in itself we convert it into a runnable using RunnableParallel for parallel execution
chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x["short"]) | short | llm | output_parser, 
    "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed | llm | output_parser
})

# result = chain.invoke({"topic":"Maths"})

#We can also give different inputs to different runnables using dictionary
result = chain.invoke({"short":{"topic":"School"}, "detailed":{"topic":"Jobs"}})


print(result)

