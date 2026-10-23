import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Open the vault
load_dotenv()

# 2. Set the AI "Dials"
llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.4
)

# 3. The Intentional Request
print("--- REQUESTING STARTUP IDEA ---")
response = llm.invoke(
    "Give me one startup idea for a 12-year-old developer."
)

# 4. Show the response content
print(f"AI Response: {response.content}")