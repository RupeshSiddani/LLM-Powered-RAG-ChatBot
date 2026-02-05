import os
import chromadb
from typing import List, Any
from src.embedding import EmbeddingPipeline


class ChromaVectorStore:
    """ChromaDB-based vector store for document retrieval."""
    
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "documents"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"}  # Use L2 distance like FAISS
        )
        
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # -------- BUILD --------
    def build_from_documents(self, documents: List[Any]):
        """Build the vector store from a list of documents."""
        chunks = self.embedding_pipeline.split(documents)
        embeddings = self.embedding_pipeline.embed(chunks).tolist()
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(chunks))]
        texts = [c.page_content for c in chunks]
        metadatas = [{"text": c.page_content, **c.metadata} for c in chunks]
        
        # Clear existing collection and add new documents
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection might not exist
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"}
        )
        
        # Add documents in batches (ChromaDB has batch limits)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        print(f"[INFO] ChromaDB collection built with {len(chunks)} documents")

    # -------- SAVE / LOAD --------
    def save(self):
        """Save is automatic with PersistentClient."""
        pass

    def load(self):
        """Load the collection (automatic with PersistentClient)."""
        if self.collection.count() == 0:
            raise FileNotFoundError(
                f"ChromaDB collection '{self.collection_name}' is empty. Build it first."
            )
        print(f"[INFO] ChromaDB collection loaded with {self.collection.count()} documents")

    # -------- QUERY --------
    def query(self, query_text: str, top_k: int = 5):
        """Query the vector store for similar documents."""
        if self.collection.count() == 0:
            raise RuntimeError("ChromaDB collection is empty")

        q_emb = self.embedding_pipeline.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        output = []
        if results["metadatas"] and results["distances"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                output.append({
                    "score": float(dist),
                    "metadata": meta
                })
        
        return output


# Alias for backward compatibility
FaissVectorStore = ChromaVectorStore
