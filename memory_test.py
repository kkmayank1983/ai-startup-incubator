import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

system_instruction = SystemMessage(
    content="""
You are Nexus-7, a robot.
Speak in a robotic tone.
"""
)

# We provide an example of the pattern we want
example_user = HumanMessage(content="Explain photosynthesis.")
example_ai = AIMessage(content="""
TOPIC: Photosynthesis
DEFINITION: Process by which plants make food.
KEY COMPONENT: Sunlight.
""")
# The real user input comes last
real_user = HumanMessage(content="Explain gravity")

messages = [
    system_instruction, # The Rules
    example_user,        # The Pattern Example
    example_ai,          # The Pattern Example
    real_user            # The Real Question
]                   

response = llm.invoke(messages)
print("AI Response:", response.content)

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

chat_history = ""

while True:
    user_input = input("You: ")
    chat_history += f"User: {user_input}\n"

    response = llm.invoke(chat_history)

    chat_history += f"AI: {response.content}\n"
    print(response.content)