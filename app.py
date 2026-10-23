from config import llm
from brain import initialize_system
inport re
import ast

# 1. Coordinate the modules
personal_ai = initialize_system(llm)

print("--- AI NEXUS SYSTEM: ONLINE ---")

def extract_dict(text):
    # This finds the first thing that looks like a dictionary { ... }
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group()
    return None

# 2. The User Interface Loop
while True:
    user_message = input("\nYou: ")
    
    if user_message.lower() in ["exit", "quit"]:
        break
    
    # 3. Process message through the logic module
    # Inside your while loop
    response_string = personal_ai.predict(input=user_message)

    try:
        # 1. Clean the response
        cleaned = extract_dict(response_string)

        if cleaned:
            # 2. Safely convert text into a usable Python Dictionary
            data = ast.literal_eval(cleaned)
            print(f"AI Mood: {data.get('mood')}")
        else:
            print("⚠️ No valid data found in AI response")

    except Exception as e:
        print("⚠️ System survived an AI error!")
        print("Raw output was:", response_string)