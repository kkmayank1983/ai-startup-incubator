import sys
try:
    import openai
    import langchain
    ai_status = "INSTALLED & READY"
except ImportError:
    ai_status = "MISSING (Run: pip install openai langchain)"

print(f"--- AI NEXUS SYSTEM CHECK ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"AI Libraries: {ai_status}")
print("Status: Virtual Lab is ACTIVE")
print("Ready for development.")