from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer expert'),
    MessagesPlaceholder(variable_name='chat_history'), #store previous context /messages
    ('human','Explain in simple terms,{query}')
])

chat_historty = []
with open('chat_history.txt') as f:
    chat_historty.append(f.readlines())
    

print(chat_historty)

# chat_template.invoke({'chat_history':chat_historty,'query':HumanMessage(content='Where is my refund?')})

prompt = chat_template.invoke({'chat_history':chat_historty,'query':'Where is my refund?'})