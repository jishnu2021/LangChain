from  langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnableMap, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()   
parser = StrOutputParser()

gen_joke = RunnableSequence(prompt,model,parser)

# def word_count(text):
#     return len(text.split())

# parallel_chain = RunnableParallel({
#     "joke":RunnablePassthrough(),
#     "word_count":RunnableLambda(word_count)  
# })

parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(lambda x: len(x.split()))  
})

final_chain = RunnableSequence(gen_joke,parallel_chain)

print(final_chain.invoke({"topic":"dogs"}))
