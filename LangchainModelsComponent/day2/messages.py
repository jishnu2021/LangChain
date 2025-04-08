from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, max_output_tokens=1000)

messages =[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Machine Learning"),

]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print("AI response : ", messages)