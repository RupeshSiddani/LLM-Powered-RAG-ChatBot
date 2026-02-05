"""
FastAPI Backend for RAG Chatbot
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.embedding import EmbeddingPipeline

load_dotenv()

# Global state
vectorstore = None
rag_initialized = False

# Configuration
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global vectorstore
    vectorstore = ChromaVectorStore(PERSIST_DIR, EMBEDDING_MODEL)
    
    # Check if collection exists
    if vectorstore.collection.count() > 0:
        print(f"[INFO] Loaded existing ChromaDB collection with {vectorstore.collection.count()} documents")
        global rag_initialized
        rag_initialized = True
    
    yield
    # Cleanup on shutdown


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>RAG Chatbot API</h1><p>Frontend not found. Please create static/index.html</p>"


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": rag_initialized,
        "document_count": vectorstore.collection.count() if vectorstore else 0
    }


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    global rag_initialized, vectorstore
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temp directory for uploaded files
    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    
    try:
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file.filename)
        
        # Load documents
        documents = load_all_documents(temp_dir)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found in uploaded files")
        
        # Build vector store
        vectorstore.build_from_documents(documents)
        rag_initialized = True
        
        return {
            "status": "success",
            "files_processed": saved_files,
            "chunks_created": vectorstore.collection.count()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with streaming response."""
    global vectorstore, rag_initialized
    
    if not rag_initialized:
        raise HTTPException(status_code=400, detail="No documents loaded. Please upload documents first.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Search for relevant documents
        results = vectorstore.query(request.query, top_k=request.top_k)
        
        if not results:
            return ChatResponse(response="No relevant documents found for your query.", sources=[])
        
        # Extract text content
        texts = []
        sources = []
        for r in results:
            text = r.get("metadata", {}).get("text", "").strip()
            if text:
                texts.append(text)
                sources.append({
                    "score": r.get("score", 0),
                    "preview": text[:200] + "..." if len(text) > 200 else text
                })
        
        if not texts:
            return ChatResponse(response="No relevant content found in documents.", sources=[])
        
        # Format context
        context = "\n\n".join(f"[Document {i+1}]\n{text}" for i, text in enumerate(texts))
        
        # Generate response using Groq
        from langchain_groq import ChatGroq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        prompt = f"""You are a helpful AI assistant. Based on the following documents, answer the user's question concisely and accurately.

Documents:
{context}

Question: {request.query}

Answer:"""
        
        response = llm.invoke(prompt)
        
        return ChatResponse(
            response=response.content.strip(),
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Chat endpoint with Server-Sent Events for streaming."""
    global vectorstore, rag_initialized
    
    if not rag_initialized:
        raise HTTPException(status_code=400, detail="No documents loaded. Please upload documents first.")
    
    async def generate():
        try:
            # Search for relevant documents
            results = vectorstore.query(request.query, top_k=request.top_k)
            
            if not results:
                yield "data: No relevant documents found.\n\n"
                return
            
            # Extract text content
            texts = [r.get("metadata", {}).get("text", "").strip() for r in results if r.get("metadata", {}).get("text")]
            
            if not texts:
                yield "data: No relevant content found.\n\n"
                return
            
            context = "\n\n".join(f"[Document {i+1}]\n{text}" for i, text in enumerate(texts))
            
            # Generate streaming response
            from langchain_groq import ChatGroq
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=LLM_MODEL,
                temperature=0.1,
                max_tokens=1024,
                streaming=True
            )
            
            prompt = f"""You are a helpful AI assistant. Based on the following documents, answer the user's question concisely and accurately.

Documents:
{context}

Question: {request.query}

Answer:"""
            
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
