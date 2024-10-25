from langchain_ollama.llms import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
        ("human", "tell me a joke"),
    ]
)

messages = chat_template.format_messages(name="Bob", user_input="What is your name?")

print(messages)

model = OllamaLLM(model="nemotron-mini:latest")

response = model.invoke(messages)

print(response)
