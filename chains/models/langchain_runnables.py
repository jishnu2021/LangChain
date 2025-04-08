from  langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnableMap, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

print(chain.invoke({"topic":"dogs"}))