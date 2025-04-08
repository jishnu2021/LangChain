from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chatmodel_val = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=256)

result = chatmodel_val.invoke("What is the capital of India")

print(result.content)