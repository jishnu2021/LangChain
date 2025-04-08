from  langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnableMap, RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableBranch

load_dotenv()

prompt = PromptTemplate(
    template="Write a detail report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="write the {text} shortly and concisely",
    input_variables=["text"]
)

model = ChatOpenAI()
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt,model,parser)

branch_chain = RunnableBranch(
    # If the report is more than 2000 characters, summarize it. Otherwise, just return the report.
    (lambda x: len(x.split())>200,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

print(final_chain.invoke({"topic":"AI"}))