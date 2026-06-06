def format_chat_history(messages, limit=6):
    """
    Formats the last 'limit' messages from Streamlit's session state
    into a readable conversation history block for the LLM.
    """
    history = ""
    # Limit the memory size to avoid running out of context tokens
    recent_messages = messages[-limit:] if len(messages) > limit else messages
    
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history
