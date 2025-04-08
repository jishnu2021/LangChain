import time
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from google.api_core.exceptions import ResourceExhausted  # Import rate limit exception

# Load environment variables
load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define output parser for structured response
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Define sentiment classification prompt
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text into positive or negative:\n\n {feedback} \n\n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

# Create classification chain
classify_Chain = prompt1 | model | parser2

# Define prompts for responses based on sentiment
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

# Define branching logic based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | StrOutputParser()),
    (lambda x: x.sentiment == "negative", prompt3 | model | StrOutputParser()),
    RunnableLambda(lambda x: "Could not classify the feedback")
)

# Combine chains
chain = classify_Chain | branch_chain
print(chain.invoke({'feedback': 'This is a beautiful phone'}))





# Function to handle rate limits using exponential backoff
# def invoke_with_retries(input_data, max_retries=5):
#     retries = 0
#     while retries < max_retries:
#         try:
#             return chain.invoke(input_data)
#         except ResourceExhausted as e:
#             wait_time = (2 ** retries) + random.uniform(0, 1)  # Exponential backoff
#             print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
#             time.sleep(wait_time)
#             retries += 1
#     print("Max retries reached. Exiting.")
#     return None

# # Test the chain with error handling
# result = invoke_with_retries({"feedback": "The product is amazing and works perfectly!"})
# if result:
#     print(result)

# # Optional: Visualize the chain graph
# try:
#     chain.get_graph().print_ascii()
# except Exception as e:
#     print(f"Error generating graph: {e}")
