import os
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import json

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from supabase import Client, create_client, PostgrestAPIResponse

class SupabaseVectorStore(VectorStore):
    """Vector store that uses Supabase's pgvector extension."""
    
    def __init__(
        self,
        client: Client,
        embedding_function: Embeddings,
        collection_name: str,
        table_name: str = "documents"
    ):
        self.client = client
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.table_name = table_name

    @classmethod
    def initialize(
        cls,
        supabase_url: str = None,
        supabase_key: str = None,
        table_name: str = "documents"
    ) -> Client:
        """Initialize Supabase client and create table if it doesn't exist."""
        url = supabase_url or os.getenv("SUPABASE_URL")
        key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("Supabase URL and key must be provided")
        
        client = create_client(url, key)
        
        # Note: The pgvector extension and table should be created in the Supabase dashboard
        # or using the SQL editor. We'll just verify the table exists by attempting a query.
        try:
            result = client.table(table_name).select("id").limit(1).execute()
            if hasattr(result, 'error') and result.error is not None:
                print(f"Warning: Table {table_name} may not exist. Please ensure it's created in the Supabase dashboard.")
        except Exception as e:
            print(f"Warning: Could not verify table {table_name}. Error: {str(e)}")
        
        return client

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[Client] = None,
        collection_name: str = "default",
        table_name: str = "documents",
        **kwargs: Any,
    ) -> "SupabaseVectorStore":
        """Create a SupabaseVectorStore from texts."""
        if client is None:
            client = cls.initialize()
        
        store = cls(client, embedding, collection_name, table_name)
        store.add_texts(texts, metadatas, **kwargs)
        return store

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings in smaller batches to avoid timeouts
        batch_size = 20  # OpenAI's recommended batch size
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_function.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Insert documents with retries
        max_retries = 3
        retry_delay = 1  # seconds
        
        data = [
            {
                "id": id_,
                "content": text,
                "metadata": metadata,
                "embedding": embedding,
                "collection_name": self.collection_name,
            }
            for id_, text, metadata, embedding in zip(ids, texts, metadatas, all_embeddings)
        ]
        
        for attempt in range(max_retries):
            try:
                # Use upsert instead of insert to handle potential duplicates
                result = self.client.table(self.table_name).upsert(
                    data,
                    on_conflict="id"  # Use ID as the conflict resolution key
                ).execute()
                
                if hasattr(result, 'error') and result.error is not None:
                    raise ValueError(f"Error inserting documents: {result.error}")
                
                return ids
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise ValueError(f"Failed to insert documents after {max_retries} attempts: {str(e)}")
                import time
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query along with scores."""
        embedding = self.embedding_function.embed_query(query)
        
        # Use the match_documents RPC function
        response = self.client.rpc(
            'match_documents',
            {
                'query_embedding': embedding,
                'match_count': k * 3,  # Get more results initially like Chroma did
                'collection_name': self.collection_name
            }
        ).execute()

        if len(response.data) == 0:
            return []

        documents = []
        for row in response.data:
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Create Document object with metadata
            doc = Document(
                page_content=row['content'],
                metadata=metadata
            )
            score = 1.0 - row['similarity']  # Convert to distance
            documents.append((doc, score))
        
        # Sort by similarity and return top k
        documents.sort(key=lambda x: x[1])
        return documents[:k]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to query."""
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def delete_collection(self) -> None:
        """Delete all documents in the collection."""
        result = self.client.table(self.table_name).delete().eq(
            "collection_name", self.collection_name
        ).execute()
        
        if hasattr(result, 'error') and result.error is not None:
            raise ValueError(f"Error deleting collection: {result.error}")

    def count_documents(self) -> int:
        """Count unique source documents in the collection."""
        all_records = []
        page_size = 1000
        offset = 0
        
        # First, get total count using count() function
        total_count = self.client.table(self.table_name).select(
            "id", 
            count="exact"
        ).eq(
            "collection_name", self.collection_name
        ).execute()
        
        print(f"Total records in collection: {total_count.count}")
        
        # Then fetch all metadata in pages
        while True:
            result = self.client.table(self.table_name).select(
                "metadata"
            ).eq(
                "collection_name", self.collection_name
            ).range(offset, offset + page_size - 1).execute()
            
            if not result.data:
                break
                
            all_records.extend(result.data)
            offset += len(result.data)
            print(f"Fetched {len(all_records)} records so far...")
            
            if len(result.data) < page_size:
                break
        
        # Count unique URLs
        unique_urls = set()
        for row in all_records:
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    continue
            
            url = metadata.get("url") if isinstance(metadata, dict) else None
            if url:
                unique_urls.add(url)
        
        print(f"Total records processed: {len(all_records)}")
        print(f"Unique URLs found: {len(unique_urls)}")
        return len(unique_urls) 