from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of WestBengal",
    "Mumbai is the capital of Maharashtra",
]

result = model.embed_documents(documents)

print(str(result))