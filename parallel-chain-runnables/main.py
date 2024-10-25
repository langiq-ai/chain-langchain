import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

# Configure logging to capture all levels of log messages
#logging.basicConfig(level=logging.DEBUG)

# Initialize the OllamaLLM model with the specified version and parameters
model = OllamaLLM(model="nemotron-mini:latest", temperature=0.9, num_ctx=2048, mirostat_tau=0.9)

# Define the prompt template for the initial user input
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an expert in review"),
        ("user", "you are going to print our 5 features of {product} about 25 words each"),
    ]
)

# Function to analyze pros of the product
def analyze_pros(features):
    # Define the prompt template for analyzing pros
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("human", "you are going to print our 5 pros of {features} about 25 words each"),
        ]
    )
    # Format the prompt with the provided features
    return pros_template.format_prompt(features=features)

# Function to analyze cons of the product
def analyze_cons(features):
    # Define the prompt template for analyzing cons
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("human", "you are going to print our 5 cons of {features} about 25 words each"),
        ]
    )
    # Format the prompt with the provided features
    return cons_template.format_prompt(features=features)

# Function to combine pros and cons into a single string
def combine_pros_cons(pros, cons):
    return f"Pros: {pros} \n\n Cons: {cons}"

# Define the chain for processing pros
pro_chain = (RunnableLambda(analyze_pros) | model | StrOutputParser())
# Define the chain for processing cons
con_chain = (RunnableLambda(analyze_cons) | model | StrOutputParser())

# Define the main chain that processes the initial prompt and combines pros and cons
chain = (
    prompt_template |
    model |
    StrOutputParser() |
    RunnableParallel(
        branches={
            "pros": pro_chain,
            "cons": con_chain
        },
    ) |
    RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Invoke the chain with a sample product and print the response
response = chain.invoke({"product": "iphone 13"})
print(response)