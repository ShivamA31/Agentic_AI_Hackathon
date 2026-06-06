import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Cache embeddings in memory to avoid reloading them multiple times
_embeddings = None

def get_embeddings():
    """
    Initializes and returns the HuggingFace local embedding model.
    """
    global _embeddings
    if _embeddings is None:
        # Using a small and fast local model ideal for CPU-based hackathons
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def process_pdf(uploaded_file):
    """
    Saves an uploaded PDF to a temporary file, chunks its text,
    creates local embeddings, and stores them in a FAISS vector store.
    """
    # Create a temporary file to load with PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load and parse the PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Chunk the text into manageable pieces
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)
        
        # Build vector store
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def query_rag(vectorstore, query, llm):
    """
    Retrieves relevant document chunks and uses the LLM to formulate an answer.
    Returns the answer and a list of citation dictionaries.
    """
    # Retrieve top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    if not docs:
        return "I could not find any relevant information in the document.", []
        
    # Compile context and citations
    context_text = ""
    citations = []
    
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", 0) + 1  # 0-indexed page to 1-indexed page
        context_text += f"\n[Source {i+1} - Page {page}]:\n{doc.page_content}\n"
        
        citations.append({
            "index": i + 1,
            "page": page,
            "snippet": doc.page_content[:200] + "..."
        })
        
    prompt = f"""You are an AI assistant answering questions based on the provided document context.
Your goal is to answer the user's question accurately using ONLY the information provided in the context.
If the context does not contain the answer, say that you cannot find it in the document.

Context:
{context_text}

Question:
{query}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content, citations
