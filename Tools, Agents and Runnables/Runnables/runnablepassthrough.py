from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model="mistral-small-2603")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_template("You are a code generator. Generate code for {topic}")

explain_prompt = ChatPromptTemplate.from_template('''
You are a helpful assistant who explains code in simple terms.
Explain the following code in simple terms:\n
{code}
'''
)

seq1 = code_prompt | model | parser


#RunnablePassThrough returns input passed to it
#here we are passing the code output of seq1 as a input to seq2(which is a parallel runnable)
#The output of seq1 is passed to both parallels i.e. "code" and "explaination"
#The "code" returns the input itself (due to RunnablePassThrough) and the explaination runs the explain_prompt sequence

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explaination" : explain_prompt | model | parser
})

chain = seq1 | seq2

result = chain.invoke({"topic":"Palindrome in python"})

print(result['code'])
print(result['explaination'])