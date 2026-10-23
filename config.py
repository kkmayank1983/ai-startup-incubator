import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load vault variables
load_dotenv()

# Centralized LLM Configuration
# Changing settings here updates the whole system!
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7
)