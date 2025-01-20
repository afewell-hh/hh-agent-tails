import os
from typing import Dict, List, Optional
from datetime import datetime

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from .vector_store import SupabaseVectorStore

class CollectionManager:
    """Manages document collections using Supabase vector store."""
    
    def __init__(self, embedding_function: Optional[Embeddings] = None):
        """Initialize the collection manager."""
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        self.collections: Dict[str, SupabaseVectorStore] = {}
        self.supabase_client = None
        self.initialize_supabase()
        self.load_existing_collections()
    
    def initialize_supabase(self):
        """Initialize Supabase client and create necessary tables."""
        self.supabase_client = SupabaseVectorStore.initialize()
    
    def load_existing_collections(self):
        """Load existing collections from Supabase."""
        try:
            # Query distinct collection names
            result = self.supabase_client.table("documents").select(
                "collection_name"
            ).execute()
            
            # Create vector store instances for each collection
            for row in result.data:
                collection_name = row["collection_name"]
                if collection_name not in self.collections:
                    self.collections[collection_name] = SupabaseVectorStore(
                        client=self.supabase_client,
                        embedding_function=self.embedding_function,
                        collection_name=collection_name
                    )
            
            print(f"Loaded {len(self.collections)} existing collections")
        except Exception as e:
            print(f"Error loading existing collections: {str(e)}")
    
    def create_collection(self, name: str) -> SupabaseVectorStore:
        """Create a new collection."""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists")
        
        collection = SupabaseVectorStore(
            client=self.supabase_client,
            embedding_function=self.embedding_function,
            collection_name=name
        )
        
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> SupabaseVectorStore:
        """Get a collection by name."""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")
        return self.collections[name]
    
    def delete_collection(self, name: str):
        """Delete a collection."""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        collection = self.collections[name]
        collection.delete_collection()
        del self.collections[name]
    
    def clear_collection(self, name: str):
        """Clear all documents from a collection."""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")
        
        collection = self.collections[name]
        collection.delete_collection()
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        return list(self.collections.keys())
    
    def add_texts_to_collection(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Add texts to a collection."""
        collection = self.get_collection(collection_name)
        
        if metadatas is None:
            metadatas = [{
                "source": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            } for _ in texts]
        
        if len(metadatas) != len(texts):
            raise ValueError("Number of metadata items must match number of texts")
        
        return collection.add_texts(texts, metadatas) 