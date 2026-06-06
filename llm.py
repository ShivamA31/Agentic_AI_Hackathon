import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """
    Initializes the LLM based on the key present in environment variables.
    Supports both Groq API keys (gsk_...) and xAI API keys (xai-...).
    """
    # Prefer XAI_API_KEY if present, fallback to GROQ_API_KEY
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("API Key missing! Please set XAI_API_KEY or GROQ_API_KEY in your .env file.")
        
    api_key = api_key.strip()
    
    if api_key.startswith("gsk_"):
        # Initialize Groq client
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0.2
            )
        except ImportError:
            # Fallback to OpenAI API format in case langchain-groq fails to import
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.2
            )
            
    elif api_key.startswith("xai-"):
        # Initialize xAI client
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="grok-beta",
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            temperature=0.2
        )
    else:
        # Generic OpenAI API fallback
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            temperature=0.2
        )
