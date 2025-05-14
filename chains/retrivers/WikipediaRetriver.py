from langchain_community.retrievers import WikipediaRetriever


retriver = WikipediaRetriever(top_k_results=5, language="en")
query = "Python programming language"

docs = retriver.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print("\n")