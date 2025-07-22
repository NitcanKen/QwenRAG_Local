# qwen_local_rag_agent.py_Review

## Overview
This document provides a comprehensive code review of the `qwen_local_rag_agent.py` file, which implements a Streamlit-based RAG (Retrieval-Augmented Generation) application with local AI models.

## Step 1: Identifying the Purpose of Each Code Section

### **Core Purpose**
This is a Streamlit web application that creates a local RAG system allowing users to:
- Chat with AI models (Qwen3, Gemma3, DeepSeek) running locally via Ollama
- Upload PDF documents or scrape web pages for context
- Store and retrieve documents using Qdrant vector database
- Optionally search the web when local documents aren't sufficient

### **Key Components:**

1. **Imports & Setup** (Lines 1-13)
2. **Custom Embedding Class** (Lines 16-31)
3. **Streamlit UI Configuration** (Lines 34-85)
4. **Database & Document Processing Functions** (Lines 88-180)
5. **Agent Configuration** (Lines 183-220)
6. **Document Relevance & Search Logic** (Lines 223-235)
7. **Main Chat Interface** (Lines 237-400)

## Step 2: Explaining Each Line and Function

### **Import Section (Lines 1-13)**
```python
import os
import tempfile
from datetime import datetime
from typing import List
import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.embedder.ollama import OllamaEmbedder
```
**Purpose**: Imports necessary libraries for:
- File handling (`os`, `tempfile`)
- Web scraping (`bs4`)
- AI agents (`agno`)
- Document processing (`langchain`)
- Vector database (`qdrant`)
- Web interface (`streamlit`)

### **Custom Embedding Class (Lines 16-31)**
```python
class OllamaEmbedderr(Embeddings):
    def __init__(self, model_name="snowflake-arctic-embed"):
        """
        Initialize the OllamaEmbedderr with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
        """
        self.embedder = OllamaEmbedder(id=model_name, dimensions=1024)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.get_embedding(text)
```
**Purpose**: Creates a wrapper around Ollama's embedding model to make it compatible with LangChain's interface. The `snowflake-arctic-embed` model converts text into 1024-dimensional vectors.

**Key Features**:
- Implements LangChain's `Embeddings` interface
- Uses 1024-dimensional embeddings
- Provides methods for both single queries and document batches

### **Constants and UI Initialization (Lines 34-54)**
```python
# Constants
COLLECTION_NAME = "test-qwen-r1"

# Streamlit App Initialization
st.title("Qwen 3 Local RAG Reasoning Agent")

# --- Add Model Info Boxes --- 
st.info("**Qwen3:** The latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models.")
st.info("**Gemma 3:** These models are multimodal—processing text and images—and feature a 128K context window with support for over 140 languages.")
```
**Purpose**: 
- Defines the Qdrant collection name for consistency
- Sets up the Streamlit page title and information boxes
- Provides user-friendly descriptions of available models

### **Session State Management (Lines 55-77)**
```python
if 'model_version' not in st.session_state:
    st.session_state.model_version = "qwen3:1.7b"  # Default to lighter model
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True  # RAG is enabled by default
```
**Purpose**: Streamlit session state preserves variables across user interactions. This initializes default values for:
- Model selection
- Chat history
- API keys
- Configuration settings
- Document tracking

### **Sidebar Configuration (Lines 80-128)**
```python
st.sidebar.header("⚙️ Settings")

# Model Selection
st.sidebar.header("🧠 Model Choice")
model_help = """
- qwen3:1.7b: Lighter model (MoE)
- gemma3:1b: More capable but requires better GPU/RAM(32k context window)
- gemma3:4b: More capable and MultiModal (Vision)(128k context window)
- deepseek-r1:1.5b
- qwen3:8b: More capable but requires better GPU/RAM

Choose based on your hardware capabilities.
"""
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["qwen3:1.7b", "gemma3:1b", "gemma3:4b", "deepseek-r1:1.5b", "qwen3:8b"],
    help=model_help
)
```
**Purpose**: Creates a sidebar with configuration options:
- Model selection with helpful descriptions
- RAG mode toggle
- Search tuning parameters
- Web search configuration
- Clear chat functionality

### **Database Connection (Lines 130-140)**
```python
def init_qdrant() -> QdrantClient | None:
    """Initialize Qdrant client with local Docker setup.

    Returns:
        QdrantClient: The initialized Qdrant client if successful.
        None: If the initialization fails.
    """
    try:
        return QdrantClient(url="http://localhost:6333")
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None
```
**Purpose**: 
- Establishes connection to local Qdrant instance
- Uses Docker default port (6333)
- Provides error handling with user-friendly messages

### **Document Processing Functions**

#### **PDF Processing (Lines 143-167)**
```python
def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []
```
**Purpose**: 
1. Creates a temporary file from uploaded PDF
2. Uses PyPDFLoader to extract text
3. Adds metadata (source type, filename, timestamp)
4. Splits text into 1000-character chunks with 200-character overlap for better retrieval

#### **Web Scraping (Lines 170-194)**
```python
def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []
```
**Purpose**: 
- Uses BeautifulSoup to extract specific HTML elements (content areas)
- Avoids headers/footers/ads by targeting content-specific CSS classes
- Applies same chunking strategy as PDF processing

### **Vector Store Management (Lines 197-230)**
```python
def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=OllamaEmbedderr()
        )
        
        # Add documents
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None
```
**Purpose**: 
1. Creates a Qdrant collection configured for 1024-dimensional vectors
2. Uses cosine similarity for document matching
3. Stores processed documents as embeddings for semantic search
4. Handles collection existence gracefully

### **Agent Configuration (Lines 232-277)**

#### **Web Search Agent**
```python
def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.2"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response
        """,
        show_tool_calls=True,
        markdown=True,
    )
```

#### **RAG Agent**
```python
def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="Qwen 3 RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.

        When asked a question:
        - Analyze the question and answer the question with what you know.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        
        Always maintain high accuracy and clarity in your responses.
        """,
        show_tool_calls=True,
        markdown=True,
    )
```
**Purpose**: Creates AI agents using the Agno framework:
- **RAG Agent**: Main conversational AI using selected Ollama model
- **Web Search Agent**: Specialized for web search using Exa API

### **Document Relevance Check (Lines 280-290)**
```python
def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    if not vector_store:
        return False, []
        
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs
```
**Purpose**: 
- Checks if relevant documents exist above similarity threshold
- Returns top 5 most relevant documents
- Used to decide between document search and web search

### **Main Chat Interface (Lines 293-400)**

#### **Chat Input Setup**
```python
chat_col, toggle_col = st.columns([0.9, 0.1])

with chat_col:
    prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")

with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")
```
**Purpose**: Creates a chat interface with:
- Text input for user queries
- Toggle button to force web search
- Dynamic placeholder text based on RAG mode

#### **Document Upload Section (Lines 300-350)**
```python
with st.expander("📁 Upload Documents or URLs for RAG", expanded=False):
    if not qdrant_client:
        st.warning("⚠️ Please configure Qdrant API Key and URL in the sidebar to enable document processing.")
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True, 
            type='pdf'
        )
        url_input = st.text_input("Enter URL to scrape")
```
**Purpose**: 
- Provides file upload interface for PDFs
- Allows URL input for web scraping
- Shows processing status and prevents duplicate processing

## Step 3: Clarifying the Running Flow

### **Application Flow:**

1. **Initialization Phase**:
   ```
   User opens app → Streamlit loads → Session state initialized → UI rendered
   ```

2. **Document Upload Phase** (if RAG enabled):
   ```
   User uploads PDF/URL → process_pdf()/process_web() → Text chunking → 
   Embedding generation → Store in Qdrant → Vector store ready
   ```

3. **Query Processing Phase**:
   ```
   User asks question → Query analysis → Document retrieval OR Web search → 
   Context preparation → Agent response generation → Display result
   ```

### **Decision Tree for Query Handling:**

```
User Query Input
    ↓
Is RAG enabled?
    ├─ No → Direct chat with selected model
    └─ Yes → Is web search forced?
        ├─ Yes → Use web search
        └─ No → Search local documents
            ├─ Found relevant docs → Use document context
            └─ No relevant docs → Fallback to web search (if enabled)
```

### **Query Processing Logic (Lines 360-400)**

#### **RAG Mode Processing**
```python
if st.session_state.rag_enabled:
    # Step 1: Query evaluation
    with st.spinner("🤔Evaluating the Query..."):
        rewritten_query = prompt
    
    # Step 2: Document search
    if not st.session_state.force_web_search and st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5, 
                "score_threshold": st.session_state.similarity_threshold
            }
        )
        docs = retriever.invoke(rewritten_query)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            st.info(f"📊 Found {len(docs)} relevant documents")
    
    # Step 3: Web search fallback
    if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
        web_search_agent = get_web_search_agent()
        web_results = web_search_agent.run(rewritten_query).content
        context = f"Web Search Results:\n{web_results}"
    
    # Step 4: Generate response
    rag_agent = get_rag_agent()
    full_prompt = f"""Context: {context}
Original Question: {prompt}
Please provide a comprehensive answer based on the available information."""
    response = rag_agent.run(full_prompt)
```

#### **Simple Mode Processing**
```python
else:
    # Simple mode without RAG
    rag_agent = get_rag_agent()
    response = rag_agent.run(prompt)
    
    # Handle thinking process extraction
    import re
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response_content, re.DOTALL)
    
    if think_match:
        thinking_process = think_match.group(1).strip()
        final_response = re.sub(think_pattern, '', response_content, flags=re.DOTALL).strip()
```

### **Key Technical Decisions Explained:**

**Why Qdrant?**: Vector database optimized for similarity search, running locally via Docker.

**Why chunk overlap?**: The 200-character overlap ensures important information isn't lost at chunk boundaries.

**Why similarity threshold?**: Prevents irrelevant documents from being included; adjustable via slider.

**Why multiple models?**: Different models have different capabilities:
- `qwen3:1.7b`: Lightweight, good for basic tasks
- `gemma3:4b`: Multimodal (text + images), larger context window
- `deepseek-r1:1.5b`: Reasoning-focused model

### **Error Handling Strategy:**
The code includes try-catch blocks around:
- Database connections
- File processing
- API calls
- Model inference

This ensures the app continues running even if individual components fail.

### **Performance Considerations:**
- Documents are processed once and stored (session state tracking)
- Vector similarity search is much faster than re-processing documents
- Web search is used as fallback to avoid unnecessary API calls
- Model selection allows users to balance capability vs. resource usage

### **State Management:**
- Session state preserves chat history and configuration
- Processed documents are tracked to avoid reprocessing
- Vector store is maintained across interactions

## Conclusion

This architecture creates a flexible, locally-running RAG system that can work entirely offline (except for optional web search) while providing a user-friendly interface for document interaction. The code is well-structured with clear separation of concerns, comprehensive error handling, and thoughtful user experience design.

The application successfully combines multiple AI technologies (Ollama models, vector databases, web search) into a cohesive system that can adapt to different use cases and hardware constraints.