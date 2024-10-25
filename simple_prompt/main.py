import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Configure logging to capture all levels of log messages
#logging.basicConfig(level=logging.DEBUG)

# Log the start of the script
logging.debug("Starting the script")

# Create a chat prompt template with a series of messages
chat_template = ChatPromptTemplate.from_messages(
    [
        # System message to set the context for the AI bot
        ("system", "You are a helpful AI bot. Your name is {name}."),
        # Human message to initiate conversation
        ("human", "Hello, how are you doing?"),
        # AI response to the human message
        ("ai", "I'm doing well, thanks!"),
        # Human message to ask a question
        ("human", "{user_input}"),
        # Human message to ask for a joke
        ("human", "tell me a joke"),
    ]
)

# Log the creation of the chat template
logging.debug("Chat prompt template created")

# Format the messages with specific values
messages = chat_template.format_messages(name="Bob", user_input="What is your name?")

# Log the formatted messages
logging.debug(f"Formatted messages: {messages}")

# Print the formatted messages
print(messages)

# Initialize the OllamaLLM model with the specified version
model = OllamaLLM(model="nemotron-mini:latest")

# Log the initialization of the model
logging.debug("OllamaLLM model initialized")

# Invoke the model with the formatted messages
response = model.invoke(messages)

# Log the response from the model
logging.debug(f"Model response: {response}")

# Print the response from the model
print(response)

# Log the end of the script
logging.debug("End of the script")