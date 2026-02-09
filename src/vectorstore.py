import os
import hashlib
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
    
    def _generate_doc_id(self, content: str, source: str = "") -> str:
        """Generate unique ID based on content hash.
        Uses only filename (not full path) to handle temp directory changes.
        """
        # Use only the filename, not the full path (temp paths change each upload)
        filename = os.path.basename(source) if source else ""
        hash_input = f"{filename}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    # -------- BUILD --------
    def build_from_documents(self, documents: List[Any], clear_existing: bool = False):
        """Build the vector store from a list of documents.
        
        Args:
            documents: List of documents to add
            clear_existing: If True, clears the collection before adding (default: False)
        """
        if clear_existing:
            # Clear collection if requested
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "l2"}
                )
            except Exception:
                pass
        
        chunks = self.embedding_pipeline.split(documents)
        
        # Get existing IDs to avoid duplicates
        existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        
        # Filter out chunks that already exist
        new_chunks = []
        new_ids = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            doc_id = self._generate_doc_id(chunk.page_content, source)
            if doc_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(doc_id)
        
        if not new_chunks:
            print(f"[INFO] No new documents to add. Collection has {self.collection.count()} documents.")
            return
        
        # Generate embeddings only for new chunks
        embeddings = self.embedding_pipeline.embed(new_chunks).tolist()
        texts = [c.page_content for c in new_chunks]
        metadatas = [{"text": c.page_content, **c.metadata} for c in new_chunks]
        
        # Add documents in batches
        batch_size = 5000
        for i in range(0, len(new_ids), batch_size):
            self.collection.add(
                ids=new_ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        print(f"[INFO] Added {len(new_chunks)} new documents. Total: {self.collection.count()}")

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
    
    def get_sources(self) -> List[str]:
        """Get list of unique document sources in the collection."""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                # Extract filename from path
                source = os.path.basename(meta["source"])
                sources.add(source)
        return list(sources)
    
    def clear(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "l2"}
            )
            print("[INFO] Collection cleared")
        except Exception as e:
            print(f"[ERROR] Failed to clear collection: {e}")


# Alias for backward compatibility
FaissVectorStore = ChromaVectorStore
