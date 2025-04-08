from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = PromptTemplate(
    input_variables=["input"],
    template="Write 5 interesting fact about {input}",
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=256)


parser= StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"input": "football"}))

# chain.get_graph().print_ascii()

