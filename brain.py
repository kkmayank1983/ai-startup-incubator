from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def initialize_system(llm):
    """
    Sets up the Personality Rules 
    and Memory Notebook 
    """

    # --- Persona Rules  ---
    system_rules = """
    You are a Data Engine. 
    You MUST respond ONLY with a valid Python dictionary.
    Do NOT include any text before or after it.
    Do NOT include explanations or markdown formatting.

    Required Format:
    {{"reply": "your response", "mood": "happy/sad/bored"}}
    """

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_rules),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
    

    # --- Memory Notebook  ---
    memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)
    # --- Create the AI Brain ---
    bot = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )

    return bot