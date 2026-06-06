from modules.router import classify_query
from modules.memory import format_chat_history
from modules.web_search import web_search
from modules.rag import process_pdf, query_rag
from modules.llm import get_llm
import streamlit as st
import os
import json
import time
from dotenv import load_dotenv

# Ensure modules are importable
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AuraChat - Agentic AI Companion",
    page_icon="🤖",
    layout="wide"
)

# Custom premium styling for Hackathon WOW factor
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #c084fc, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Badge styling */
    .tool-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .badge-llm {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
    }
    
    .badge-rag {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .badge-web {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }

    /* Citation Card styling */
    .citation-card {
        background-color: #1e293b;
        border-left: 4px solid #10b981;
        padding: 10px;
        border-radius: 4px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title and Header
st.title("🤖 AuraChat")
st.markdown("<p style='font-size:1.1rem; color:#94a3b8; margin-top:-15px;'>Your hybrid agentic assistant with PDF RAG, Web Search, and Chat Memory</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration & Uploads")

    # 1. API Key Validation UI
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        st.warning("⚠️ API Key not found in `.env` file.")
        user_key = st.text_input("Enter Groq or xAI API Key:", type="password")
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
            st.success("API Key loaded temporarily!")
            st.rerun()
    else:
        st.success("🔑 API Key Loaded from Environment")

    # 2. PDF RAG Upload Section
    st.markdown("---")
    st.markdown("### 📄 Document RAG")
    uploaded_file = st.file_uploader(
        "Upload a PDF document to query:", type=["pdf"])

    if uploaded_file:
        # Avoid reloading/re-embedding on every session rerun
        if "processed_pdf_name" not in st.session_state or st.session_state.processed_pdf_name != uploaded_file.name:
            with st.spinner("Processing document chunks and creating FAISS index (local CPU)..."):
                try:
                    vectorstore = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_pdf_name = uploaded_file.name
                    st.success(f"Successfully loaded '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Failed to process PDF: {str(e)}")
    else:
        # Clear vectorstore from session if file is removed
        if "processed_pdf_name" in st.session_state:
            del st.session_state.processed_pdf_name
            del st.session_state.vectorstore

    # 3. Actions / Utility Buttons
    st.markdown("---")
    st.markdown("### 🛠️ Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        # Download Chat History logic
        if "messages" in st.session_state and st.session_state.messages:
            history_data = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="📥 Download",
                data=history_data,
                file_name="chat_history.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📥 Download", disabled=True, use_container_width=True)

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Render the custom Badge
        if msg["role"] == "assistant" and "tool" in msg:
            badge_class = "badge-llm"
            if msg["tool"] == "Document Search":
                badge_class = "badge-rag"
            elif msg["tool"] == "Web Search":
                badge_class = "badge-web"

            st.markdown(
                f'<span class="tool-badge {badge_class}">{msg["tool"]}</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # Render citations if any
        if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
            with st.expander("📚 Source Citations"):
                for cite in msg["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>Source Chunk {cite['index']} (Page {cite['page']})</strong><br/>
                        {cite['snippet']}
                    </div>
                    """, unsafe_allow_html=True)

        # Render search logs if any
        if msg["role"] == "assistant" and "search_logs" in msg and msg["search_logs"]:
            with st.expander("🌐 Raw Web Search Results"):
                st.code(msg["search_logs"])

# Helper generator for smooth typing animation


def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


# ----------------- CHAT HANDLING -----------------
if prompt := st.chat_input("Ask me anything..."):
    # Check for API key
    active_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if not active_key:
        st.error(
            "Missing API Key! Please enter your key in the sidebar or setup the `.env` file.")
        st.stop()

    # Render user query
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render assistant response container
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        badge_placeholder = st.empty()
        response_placeholder = st.empty()

        with status_placeholder.status("Thinking...", expanded=True) as status:
            try:
                # 1. Initialize LLM
                llm = get_llm()

                # 2. Query Routing
                has_pdf = "vectorstore" in st.session_state
                status.write("Determining intent routing...")
                route = classify_query(prompt, has_pdf, llm)

                # 3. Execution based on route
                final_answer = ""
                citations = []
                search_logs = ""
                tool_label = ""

                if route == "document":
                    status.write("Running Document RAG query...")
                    tool_label = "Document Search"
                    final_answer, citations = query_rag(
                        st.session_state.vectorstore, prompt, llm)

                elif route == "web_search":
                    status.write("Searching the web via DuckDuckGo...")
                    tool_label = "Web Search"
                    final_answer, search_logs = web_search(prompt, llm)

                else:
                    status.write("Invoking general knowledge LLM...")
                    tool_label = "LLM"
                    history_str = format_chat_history(
                        st.session_state.messages[:-1])

                    llm_prompt = f"""You are a helpful AI assistant. Answer the user's query utilizing the conversation history for context.

Conversation History:
{history_str}

User Question: {prompt}
Answer:"""
                    final_answer = llm.invoke(llm_prompt).content

                status.update(
                    label=f"Query answered via {tool_label}!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Error processing query!",
                              state="error", expanded=True)
                st.error(f"An error occurred: {str(e)}")
                st.stop()

        # Remove status placeholder so it looks cleaner
        status_placeholder.empty()

        # Display Tool Badge
        badge_class = "badge-llm"
        if tool_label == "Document Search":
            badge_class = "badge-rag"
        elif tool_label == "Web Search":
            badge_class = "badge-web"
        badge_placeholder.markdown(
            f'<span class="tool-badge {badge_class}">{tool_label}</span>', unsafe_allow_html=True)

        # Stream the typing animation
        response_placeholder.write_stream(stream_text(final_answer))

        # Display citations if RAG
        if citations:
            with st.expander("📚 Source Citations"):
                for cite in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>Source Chunk {cite['index']} (Page {cite['page']})</strong><br/>
                        {cite['snippet']}
                    </div>
                    """, unsafe_allow_html=True)

        # Display search logs if Web Search
        if search_logs:
            with st.expander("🌐 Raw Web Search Results"):
                st.code(search_logs)

        # Save response in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "tool": tool_label,
            "citations": citations,
            "search_logs": search_logs
        })
