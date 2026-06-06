def classify_query(query, has_pdf, llm):
    """
    Classifies the user query into: 'document', 'web_search', or 'general'.
    Uses rule-based checks first for speed and reliability, and falls back
    to LLM-based classification to give it an agentic router feel.
    """
    query_lower = query.lower()
    
    # 1. Rule-based override for Web Search
    # Check for keywords that strongly signal a need for current web info
    web_keywords = ["today", "latest", "news", "current", "weather", "recent", "now", "live search"]
    if any(kw in query_lower for kw in web_keywords):
        return "web_search"
        
    # 2. Rule-based override for Document (if a PDF is uploaded)
    # Check for words referencing the loaded document
    doc_keywords = ["document", "pdf", "file", "uploaded", "doc", "paper", "textbook", "in this", "refer to"]
    if has_pdf and any(kw in query_lower for kw in doc_keywords):
        return "document"
        
    # 3. LLM-based Classification (Agent Router)
    if not has_pdf:
        # Without a PDF, we only need to route between general chat and web search
        prompt = f"""You are a query classifier router. Classify the user query into either 'web_search' or 'general'.
Use 'web_search' if the query asks about current events, news, or requires real-time information.
Otherwise, use 'general' for general chat, coding, general knowledge, math, etc.

Output ONLY the category name ('web_search' or 'general'). Do not explain.

Query: "{query}"
Category:"""
        try:
            response = llm.invoke(prompt)
            decision = response.content.strip().lower()
            if "web_search" in decision:
                return "web_search"
        except Exception:
            pass
        return "general"
        
    # With a PDF, we choose between document, web_search, and general
    prompt = f"""You are a query classifier router. Classify the user query into one of three categories: 'document', 'web_search', or 'general'.
- 'document': User is asking about the uploaded document, its content, or summary.
- 'web_search': User is asking about current/live events, recent news, or real-time info.
- 'general': General chat, math, coding, or general questions not related to the uploaded document or current events.

Output ONLY the category name ('document', 'web_search', or 'general'). Do not explain.

Query: "{query}"
Category:"""
    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "document" in decision:
            return "document"
        elif "web_search" in decision:
            return "web_search"
    except Exception:
        pass
    
    return "general"
