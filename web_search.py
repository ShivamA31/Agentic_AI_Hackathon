from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def web_search(query, llm):
    """
    Performs a DuckDuckGo web search, handles errors gracefully,
    and asks the LLM to summarize/compile the final answer.
    """
    try:
        # Initialize DuckDuckGo search API wrapper and tool
        # Setting max_results to 3 or 4 keeps context size reasonable
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
        search = DuckDuckGoSearchRun(api_wrapper=wrapper)
        search_results = search.run(query)
    except Exception as e:
        search_results = f"DuckDuckGo search failed or rate-limited: {str(e)}"

    # Generate answer summarizing web results
    prompt = f"""You are an AI assistant tasked with answering questions using search results.
Please summarize the following search results to provide a comprehensive, fact-based answer.
If the results don't contain enough information to answer, state that.

Search Results:
{search_results}

Question:
{query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content, search_results
