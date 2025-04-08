import random

class NakliLLM:
    
    def __init__(self):
        print("Initializing LLM..")
    
    
    def predict(self,prompt):
        response_list=[
            'Delhi is the capital of India.',
            'IPL is a cricket tournament.',
            'India is a country in South Asia.',
        ]
        return {'response': random.choice(response_list)}

llm = NakliLLM()

class NakliPromptTemplate:
    
    def __init__(self, template,input_variables):
        self.input_variables = input_variables
        self.template = template
    
    def format(self, input_dict):
        return self.template.format(**input_dict)
    
    
template = NakliPromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

fromated_output = template.format({"topic":"India"})

print(llm.predict(fromated_output))



class NakliLLMChain:
    
    def __init__(self,prompt, llm):
        self.prompt = prompt
        self.llm = llm
        print("LLM Chain initialized")
        
    def run(self,input_dict):
        final_prompt = self.prompt.format(input_dict)
        final_response = self.llm.predict(final_prompt)
        return final_response['response']


chain = NakliLLMChain(prompt=template, llm=llm)
response = chain.run({"topic":"India"}) 
print(response)