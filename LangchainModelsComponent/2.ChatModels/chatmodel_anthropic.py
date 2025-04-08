from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

chatmodel_val = ChatAnthropic(temperature=0.7, model="claude-3-5-sonnet-20241022", max_tokens=2000)

result = chatmodel_val.invoke("What is the capital of France?")

print(result.content)