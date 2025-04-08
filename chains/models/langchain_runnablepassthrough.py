from  langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnableMap, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableParallel

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


joke_chain = RunnableSequence(prompt,model,parser)

parallel_Chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2,model,parser)    
})

final_chain = RunnableSequence(joke_chain,parallel_Chain)

print(final_chain.invoke({"topic":"cricket"}))