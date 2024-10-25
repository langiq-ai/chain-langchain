import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

# Configure logging to capture all levels of log messages
#logging.basicConfig(level=logging.DEBUG)

# Initialize the OllamaLLM model with the specified version and parameters
model = OllamaLLM(model="nemotron-mini:latest", temperature=0.9, num_ctx=2048, mirostat_tau=0.9)

# Define the prompt template for the initial user input
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a customer feedback analysis agent"),
        ("user", "based on the {feedbacks}, you are going to print out positive, negative, neutral or escalation feedbacks"),
    ]
)

# Define the prompt templates for different types of feedback
positive_feedbacks_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful agent"),
        ("user", "Generate a thank you message based on this feedback {feedbacks}"),
    ]
)
negative_feedbacks_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful agent"),
        ("user", "Generate a 'we will get better' message based on this feedback {feedbacks}"),
    ]
)
neutral_feedbacks_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful agent"),
        ("user", "Generate a thank you message and 'we will improve to meet and exceed your expectation' based on this feedback {feedbacks}"),
    ]
)
escalation_feedbacks_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful agent"),
        ("user", "Generate 'I will have a human contact you' based on this {feedbacks}"),
    ]
)

# Define the prompt template for classifying feedback sentiment
classification_feedbacks_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful agent"),
        ("user", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Define the branch logic for handling different types of feedback
branch = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedbacks_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedbacks_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedbacks_template | model | StrOutputParser()
    ),
    escalation_feedbacks_template | model | StrOutputParser()
)

# Define the chain for classifying feedback
classification_chain = classification_feedbacks_template | model | StrOutputParser()

# Combine the classification chain with the branch logic
chain = classification_chain | branch

# Sample feedback for testing
review = "it worked, thank you"

# Log the input review
logging.debug(f"Input review: {review}")

# Invoke the chain with the sample feedback and log the result
result = chain.invoke({"feedback": review})
logging.debug(f"Result: {result}")

# Print the result
print(result)