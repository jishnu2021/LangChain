from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="This is a test document.", metadata={"source": "test_source"}),
    Document(page_content="This is another test document.", metadata={"source": "test_source_2"}),
]

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding, persist_directory="chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "test document"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("\n")