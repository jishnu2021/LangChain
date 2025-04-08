from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1= ChatGoogleGenerativeAI(model="gemini-1.5-pro")
model2= ChatGoogleGenerativeAI(model="gemini-1.5-pro")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the simple text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short questions from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provied text and the questions into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_Chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
})

merge_Chain = prompt3 | model1 | parser

chain = parallel_Chain | merge_Chain

text="""
AI Engineer: Role, Responsibilities, and Skills
An AI Engineer is a specialized software engineer who develops artificial intelligence (AI) systems and applications. They design, build, and optimize AI models, enabling machines to perform tasks that typically require human intelligence, such as problem-solving, decision-making, speech recognition, and image processing. AI engineers play a crucial role in various industries, including healthcare, finance, robotics, and e-commerce.

Role of an AI Engineer
AI engineers create intelligent solutions by leveraging machine learning (ML), deep learning, and natural language processing (NLP). They work on AI-driven applications such as chatbots, recommendation systems, fraud detection algorithms, and autonomous vehicles. Their primary goal is to develop and integrate AI models that enhance automation and efficiency.

Key Responsibilities
Designing AI Models: AI engineers build and optimize machine learning and deep learning models using frameworks like TensorFlow, PyTorch, and Scikit-learn.

Data Preprocessing and Analysis: They clean, transform, and analyze large datasets to improve model accuracy and efficiency.

Developing AI Algorithms: AI engineers create algorithms for pattern recognition, predictive analytics, and anomaly detection.

Deploying AI Models: They integrate AI solutions into production environments using cloud services such as AWS, Google Cloud, and Azure.

Optimizing Model Performance: AI engineers fine-tune models for better accuracy, speed, and scalability.

Collaboration with Teams: They work closely with data scientists, software developers, and business stakeholders to ensure AI solutions align with business goals.

Ensuring Ethical AI Practices: AI engineers consider bias, fairness, and transparency while developing AI systems.

Skills Required
To become a successful AI engineer, one needs a strong foundation in programming, mathematics, and machine learning. Some essential skills include:

Programming Languages: Python, Java, C++, and R

Machine Learning & Deep Learning: Knowledge of neural networks, supervised and unsupervised learning

Mathematics & Statistics: Probability, linear algebra, calculus, and optimization techniques

Big Data Technologies: Experience with Hadoop, Spark, and Kafka for handling large datasets

Cloud Computing: Familiarity with cloud platforms for AI model deployment

Natural Language Processing (NLP): Understanding text processing and speech recognition

Computer Vision: Image and video processing using deep learning

Industries and Applications
AI engineers are in high demand across various industries:

Healthcare: AI-powered diagnostics, drug discovery, and robotic surgery

Finance: Fraud detection, algorithmic trading, and credit scoring

Retail & E-commerce: Recommendation engines and customer service automation

Automotive: Self-driving cars and predictive maintenance

Cybersecurity: AI-driven threat detection and risk assessment

Career Opportunities
AI engineers can work as Machine Learning Engineers, Data Scientists, AI Researchers, NLP Engineers, or Computer Vision Engineers. With AI revolutionizing industries, the demand for AI professionals continues to grow.

Conclusion
An AI Engineer is at the forefront of technological innovation, developing intelligent systems that transform businesses and improve lives. With the right skills and expertise, AI engineers contribute to solving complex real-world problems, making AI one of the most exciting and rewarding fields in technology today.
"""
print(chain.invoke({'text': text}))

# chain.get_graph().print_ascii()