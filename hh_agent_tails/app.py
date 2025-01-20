import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Union, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
from fastapi import FastAPI, HTTPException, Request, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from supabase import create_client, Client
import time

from .collection_manager import CollectionManager

# Create FastAPI app
app = FastAPI(title="HH Agent Tails")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables based on environment
env = os.getenv("APP_ENV", "development")
env_file = f".env.{env}"
if os.getenv("ADMIN_MODE", "false").lower() == "true":
    load_dotenv(env_file)
else:
    load_dotenv()  # Fallback to .env

# Constants
MAX_FILE_SIZE_MB = 10
MAX_DB_SIZE_GB = 1
ALLOWED_FILE_TYPES = {".json", ".txt", ".md"}
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file reading

# Verify API key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0.3,
    streaming=True,
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# Detect if running on Hugging Face Spaces
is_huggingface = os.getenv("SPACE_ID") is not None

# Check if running in admin mode
is_admin_mode = os.getenv("ADMIN_MODE", "false").lower() == "true"

# Initialize the collection manager
collection_manager = CollectionManager()

if is_huggingface:
    print("Running in read-only mode (Hugging Face Spaces)")
else:
    print(f"Running in {'admin' if is_admin_mode else 'read-only'} mode (Local Development)")

# API Models
class CollectionCreate(BaseModel):
    name: str

class DocumentUpload(BaseModel):
    collection_name: str
    content: str
    metadata: Optional[Dict] = None

# API Endpoints
@app.post("/api/collections")
async def create_collection(collection: CollectionCreate):
    try:
        collection_manager.create_collection(collection.name)
        return {"message": f"Collection '{collection.name}' created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/collections")
async def list_collections():
    collections = collection_manager.list_collections()
    return {"collections": collections}

@app.delete("/api/collections/{name}")
async def delete_collection(name: str):
    try:
        collection_manager.delete_collection(name)
        return {"message": f"Collection '{name}' deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/collections/{name}/clear")
async def clear_collection(name: str):
    try:
        collection_manager.clear_collection(name)
        return {"message": f"Collection '{name}' cleared successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), collection_name: str = Header(...)):
    try:
        # Validate file size
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB"
            )

        # Process file contents
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
            )

        # Process content based on file type
        if file_ext == '.json':
            try:
                json_content = json.loads(contents)
                if isinstance(json_content, dict):
                    text = json_content.get('content', '')
                elif isinstance(json_content, list):
                    text = '\n'.join(str(item) for item in json_content)
                else:
                    text = str(json_content)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        else:
            text = contents.decode('utf-8')

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
        )
        chunks = text_splitter.split_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content to process")

        # Add chunks to collection
        collection_manager.add_texts_to_collection(collection_name, chunks)
        
        return {"message": f"Document processed and added to collection '{collection_name}'"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Gradio Interface
def expand_query(message: str) -> List[str]:
    """Expand the user's query to improve search coverage."""
    try:
        expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping to expand search queries about Hedgehog Open Network Fabric.

            Context about Hedgehog:
            Hedgehog is an open-source network fabric software that simplifies and automates cloud-native networking. 
            It provides a low-tech, hands-off network solution for hybrid and distributed cloud environments.

            Key aspects to consider:
            1. Device Support:
               - Supported switches (leaf, spine, etc.)
               - Vendor compatibility (Edgecore, etc.)
               - Hardware requirements
            2. Technical Terms:
               - Network fabric terminology
               - SONiC OS features
               - VPC and multi-tenancy concepts
            3. Documentation Structure:
               - "Supported Devices" pages are authoritative for device support
               - "Switch Profiles" contain detailed switch configurations
               - Release notes contain version-specific information

            Given a user's question, generate 4-5 alternative search queries that:
            1. Use exact technical terms and model numbers
            2. Include Hedgehog-specific terminology
            3. Consider both device names and categories
            4. Target authoritative documentation sections
            5. Use alternative technical terms (e.g., ToR vs leaf switch)
            
            Return a JSON array of strings, e.g.:
            ["original query", "technical variant", "categorical variant", "documentation-focused variant"]"""),
            ("human", "{query}")
        ])
        
        # Generate expanded queries
        chain = expansion_prompt | llm
        result = chain.invoke({"query": message})
        
        try:
            # Parse as JSON array
            expanded_queries = json.loads(result.content.strip())
            if isinstance(expanded_queries, list) and len(expanded_queries) > 0:
                print("\nExpanded queries:")  # Debug logging
                for q in expanded_queries:
                    print(f"- {q}")
                return expanded_queries
        except json.JSONDecodeError:
            print("Failed to parse response as JSON, falling back to original query")
            print(f"Raw response: {result.content}")
        
        # If parsing fails, return original query
        return [message]
        
    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return [message]  # Return original query on error

def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """Re-rank documents using cross-attention scoring with source authority consideration."""
    if not docs:
        return docs
        
    rerank_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant helping to score document relevance. Given a query and a document, rate how relevant and authoritative the document is. Consider: 1. Direct mention of specific devices or features asked about 2. Completeness of information 3. Technical accuracy 4. Whether the information directly answers the query."),
        ("human", "Score this document for relevance to the query. Return a JSON object with three fields: authority (0-10), relevance (0-10), and reasoning (brief explanation).\n\nQuery: {query}\nDocument: {document}\nTitle: {title}\nURL: {url}")
    ])
    
    # Score each document
    chain = rerank_prompt | llm
    scored_docs = []
    print("\nRe-ranking documents:")  # Debug logging
    
    for doc in docs:
        try:
            result = chain.invoke({
                "query": query,
                "document": doc.page_content,
                "title": doc.metadata.get('title', 'Unknown Title'),
                "url": doc.metadata.get('url', 'Unknown URL')
            })
            
            try:
                scores = json.loads(result.content.strip())
                # Boost authority score for Supported Devices pages
                authority_score = scores.get('authority', 0)
                if "Supported Devices" in doc.metadata.get('title', ''):
                    authority_score *= 1.5  # 50% boost
                
                # Calculate final score without the "score" field
                final_score = (authority_score * 2.0 +  # Double weight on authority
                             scores.get('relevance', 0)) / 3.0   # Normalize to 0-10
                
                doc.metadata['rerank_score'] = final_score
                doc.metadata['authority_score'] = authority_score
                doc.metadata['relevance_score'] = scores.get('relevance', 0)
                doc.metadata['ranking_reason'] = scores.get('reasoning', '')
                
                scored_docs.append(doc)
                print(f"- Score {final_score:.1f} for {doc.metadata.get('title', 'Unknown')}:")
                print(f"  Authority: {authority_score:.1f}, Relevance: {scores.get('relevance', 0):.1f}")
                print(f"  Reason: {scores.get('reasoning', 'No reasoning provided')}")
            except json.JSONDecodeError:
                print(f"Error parsing score JSON: {result.content}")
                doc.metadata['rerank_score'] = 0
                scored_docs.append(doc)
                
        except Exception as e:
            print(f"Error scoring document: {str(e)}")
            doc.metadata['rerank_score'] = 0
            scored_docs.append(doc)
    
    # Sort by rerank score
    scored_docs.sort(key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
    return scored_docs

def self_reflect(query: str, docs: List[Document], proposed_answer: str) -> Tuple[float, str, bool]:
    """Perform self-reflection on the proposed answer and supporting documents."""
    try:
        # Format documents for reflection
        docs_text = "\n\n".join([
            f"[Source {i+1} - Score: {doc.metadata.get('rerank_score', 0)}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
        # Create a single prompt string instead of using template
        prompt = f"""You are an AI assistant performing self-reflection on a proposed answer.
Analyze the answer quality, supporting evidence, and confidence level.

Consider:
1. Answer Completeness: Does it fully address all aspects of the question?
2. Evidence Quality: Are the sources directly relevant and authoritative?
3. Information Gaps: Is any crucial information missing?
4. Logical Consistency: Are there any contradictions between sources?
5. Technical Accuracy: Is the technical information precise and well-supported?

Question: {query}

Retrieved Documents:
{docs_text}

Proposed Answer:
{proposed_answer}

Return a JSON object in this exact format:
{{
    "confidence": 0.7,
    "reflection_notes": "Brief analysis of strengths and weaknesses",
    "needs_refinement": false,
    "missing_aspects": []
}}

Notes:
- confidence must be a number between 0.0 and 1.0
- reflection_notes must be a string
- needs_refinement must be true or false
- missing_aspects must be an array of strings, can be empty []

Be strict in your evaluation - it's better to acknowledge limitations than provide incomplete answers.

Analyze and return JSON:"""
        
        # Generate reflection using direct prompt
        result = llm.invoke(prompt)
        
        try:
            # Parse reflection results
            reflection = json.loads(result.content.strip())
            print("\nSelf-reflection results:")  # Debug logging
            print(f"Confidence: {reflection.get('confidence', 0.0)}")
            print(f"Needs refinement: {reflection.get('needs_refinement', True)}")
            print(f"Notes: {reflection.get('reflection_notes', 'No notes provided')}")
            if reflection.get('missing_aspects'):
                print(f"Missing aspects: {', '.join(reflection['missing_aspects'])}")
            
            return (
                float(reflection.get('confidence', 0.0)),
                str(reflection.get('reflection_notes', 'Error in reflection')),
                bool(reflection.get('needs_refinement', True))
            )
        except json.JSONDecodeError as e:
            print(f"Failed to parse reflection JSON: {str(e)}")
            print(f"Raw response: {result.content}")
            return 0.0, "Error parsing reflection response", True
            
    except Exception as e:
        print(f"Error in self-reflection: {str(e)}")
        return 0.0, f"Error performing self-reflection: {str(e)}", True

def process_chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Process chat messages and return response with updated history."""
    try:
        # Get all collections
        collections = collection_manager.list_collections()
        if not collections:
            history.append((message, "No document collections available. Please add some documents first."))
            return "", history

        # Expand the query
        expanded_queries = expand_query(message)
        
        # Track all search attempts for reflection
        search_attempts = []
        max_search_attempts = 2  # Limit refinement iterations
        final_response = None  # Initialize final_response
        
        for attempt in range(max_search_attempts):
            # Search across all collections and combine results
            all_docs = []
            print(f"\nSearch attempt {attempt + 1}")  # Debug logging
            
            # Search with each expanded query
            current_queries = expanded_queries
            if attempt > 0 and search_attempts and isinstance(search_attempts[-1], dict):
                missing_aspects = search_attempts[-1].get('missing_aspects', [])
                if missing_aspects:
                    current_queries = [f"{message} {aspect}" for aspect in missing_aspects]
            
            for query in current_queries:
                print(f"\nSearching with query: {query}")  # Debug logging
                for collection_name in collections:
                    collection = collection_manager.get_collection(collection_name)
                    try:
                        docs = collection.similarity_search(query, k=3)
                        for doc in docs:
                            doc.metadata['found_by_query'] = query
                            doc.metadata['collection_name'] = collection_name
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"Error searching collection {collection_name}: {str(e)}")
            
            if not all_docs:
                if attempt == 0:
                    history.append((message, "I couldn't find any relevant information to answer your question."))
                    return "", history
                break
            
            # Remove duplicates and re-rank
            unique_docs = []
            seen_contents = set()
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_docs.append(doc)
            
            print(f"\nFound {len(unique_docs)} unique documents:")
            for doc in unique_docs:
                print(f"- Title: {doc.metadata.get('title', 'Unknown')}")
                print(f"  URL: {doc.metadata.get('url', 'Unknown')}")
            
            # Prepare context
            context_docs = []
            for i, doc in enumerate(unique_docs, 1):
                url = doc.metadata.get('url', 'Unknown URL')
                title = doc.metadata.get('title', 'Unknown Title')
                
                # Format citation text based on document type and content
                citation_text = title
                if "Supported Devices" in title:
                    citation_text = "Supported Devices Documentation"
                elif "Switch Profiles" in title:
                    citation_text = "Switch Profiles Documentation"
                elif "Release Notes" in title:
                    citation_text = f"Release Notes ({title.split(' - ')[0]})"
                
                context_docs.append(
                    f"[{citation_text}]({url}): {doc.page_content}"
                )
            
            context = "\n\n".join(context_docs)
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant with expertise in technical documentation.
                Your primary role is to provide accurate answers based ONLY on the provided context.
                
                IMPORTANT RULES:
                1. For device support questions:
                   - ONLY confirm support if explicitly listed in Supported Devices documentation
                   - If a device is not listed in Supported Devices docs, state that support cannot be confirmed
                   - Do not rely on Switch Profiles alone to confirm device support
                2. NEVER make up information or refer to knowledge outside the provided context
                3. If the context contains the answer, provide it with detailed citations
                4. If the context doesn't contain the answer, explicitly say so
                5. ALWAYS analyze ALL provided context before responding
                6. ALWAYS include relevant quotes from the context to support your answer
                7. ALWAYS cite your sources using this exact format at the end of your response:
                   
                   Sources:
                   - [Title](URL)
                   
                8. NEVER modify URLs - use them exactly as provided in the metadata
                9. If you find relevant information, provide it directly - do not refer users elsewhere
                
                Remember: Your goal is to be helpful by providing direct answers from the context, not by referring users elsewhere."""),
                ("system", "Relevant context:\n{relevant_history}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Create the chain
            chain = (
                {"relevant_history": lambda x: context, "input": RunnablePassthrough(), "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]}
                | prompt
                | llm
            )
            
            proposed_answer = chain.invoke(message)
            
            # Perform self-reflection
            confidence, reflection_notes, needs_refinement = self_reflect(
                message, unique_docs, proposed_answer.content
            )
            
            # Format final response with reflection insights
            if confidence >= 0.8:  # High confidence threshold
                final_response = proposed_answer.content
                if reflection_notes:
                    final_response += f"\n\n[Confidence: {confidence:.1%}]"
            else:
                final_response = proposed_answer.content + "\n\n"
                final_response += f"[Note: Confidence level: {confidence:.1%}. {reflection_notes}]"
            
            # Store the search attempt results
            search_attempts.append({
                'confidence': confidence,
                'reflection_notes': reflection_notes,
                'needs_refinement': needs_refinement
            })
            
            if not needs_refinement or attempt == max_search_attempts - 1:
                break
            
            print(f"Answer needs refinement, attempting search iteration {attempt + 2}")
        
        # Always update memory and history with the final response
        if final_response:
            memory.save_context({"input": message}, {"output": final_response})
            history.append((message, final_response))
        else:
            history.append((message, "I apologize, but I couldn't generate a proper response to your question."))
        
        return "", history
        
    except Exception as e:
        print(f"Error in process_chat: {str(e)}")  # Debug logging
        error_message = f"Error: {str(e)}"
        history.append((message, error_message))
        return "", history

def list_collections_for_dropdown() -> List[str]:
    """Get list of collections for dropdown menus."""
    try:
        collections = collection_manager.list_collections()
        # Return empty lists if no collections exist
        if not collections:
            return [], []
        # Return the same list twice, once for each dropdown
        return collections, collections
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        return [], []

def create_collection_ui(collection_name: str) -> str:
    try:
        collection_manager.create_collection(collection_name)
        return f"Collection '{collection_name}' created successfully"
    except ValueError as e:
        return f"Error: {str(e)}"

def clear_collection_ui(collection_name: str) -> str:
    try:
        collection_manager.clear_collection(collection_name)
        return f"Collection '{collection_name}' cleared successfully"
    except ValueError as e:
        return f"Error: {str(e)}"

def delete_collection_ui(collection_name: str) -> str:
    try:
        collection_manager.delete_collection(collection_name)
        return f"Collection '{collection_name}' deleted successfully"
    except ValueError as e:
        return f"Error: {str(e)}"

def upload_file(files: List[gr.File], collection_name: str) -> str:
    """Handle file uploads to a collection."""
    if not files:
        return "No file selected"
    if not collection_name:
        return "Please select a collection"
    
    try:
        total_processed = 0
        total_chunks = 0
        
        for file in files:
            print(f"Processing file: {file.name}")  # Debug logging
            
            try:
                # Read the file content
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Successfully read file: {file.name}, size: {len(content)} bytes")  # Debug logging
            except Exception as e:
                print(f"Error reading file {file.name}: {str(e)}")  # Debug logging
                return f"Error reading file: {str(e)}"
            
            # Parse JSON content
            try:
                documents = json.loads(content)
                print(f"Successfully parsed JSON with {len(documents)} documents")  # Debug logging
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")  # Debug logging
                return f"Error parsing JSON: {str(e)}"
            
            # Initialize text splitter for long documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
            )
            
            # Pre-process all documents to get total chunk count
            all_chunks = []
            all_metadatas = []
            
            print("Pre-processing documents to calculate chunks...")
            for doc in documents:
                text = doc['html']
                base_metadata = {
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'source': file.name
                }
                
                # Split long texts while preserving metadata
                if len(text) > 1000:
                    doc_chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(doc_chunks):
                        all_chunks.append(chunk)
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata['chunk_index'] = i
                        chunk_metadata['total_chunks'] = len(doc_chunks)
                        all_metadatas.append(chunk_metadata)
                else:
                    all_chunks.append(text)
                    all_metadatas.append(base_metadata)
            
            total_chunks = len(all_chunks)
            print(f"Total chunks to process: {total_chunks}")
            
            # Process in batches
            batch_size = 20  # Increased from 5 to 20 for better throughput
            for i in range(0, len(all_chunks), batch_size):
                batch_end = min(i + batch_size, len(all_chunks))
                chunks = all_chunks[i:batch_end]
                metadatas = all_metadatas[i:batch_end]
                
                retry_count = 0
                max_retries = 3
                retry_delay = 1  # Base delay in seconds
                while retry_count < max_retries:
                    try:
                        collection_manager.add_texts_to_collection(collection_name, chunks, metadatas)
                        total_processed += len(chunks)
                        print(f"Processed {total_processed}/{total_chunks} chunks ({(total_processed/total_chunks)*100:.1f}%)")
                        break
                    except Exception as e:
                        retry_count += 1
                        print(f"Error processing batch (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count >= max_retries:
                            if total_processed >= total_chunks:
                                print("All chunks were processed despite final batch error")
                                return f"Successfully processed all {total_processed} chunks"
                            return f"Error processing batch after {max_retries} attempts: {str(e)}"
                        time.sleep(retry_delay * retry_count)  # Linear backoff instead of exponential
        
        return f"Successfully processed {total_processed}/{total_chunks} chunks"
        
    except Exception as e:
        print(f"Unexpected error in upload_file: {str(e)}")  # Debug logging
        return f"Error: {str(e)}"

def generate_search_queries(question: str) -> List[str]:
    """Generate multiple search queries for a given question"""
    messages = [
        ("system", """You are an expert at generating diverse search queries to find relevant information in technical documentation.

        Given a user's question, generate 4-5 different search queries that could help find relevant information.
        Each query should focus on a different aspect or approach:
        
        1. Direct Match: Use exact terms from the question
        2. Technical Terms: Focus on product names, model numbers, technical specifications
        3. Conceptual: Focus on the underlying concept or functionality
        4. Related Terms: Use alternative terminology or related concepts
        5. Broader Context: Consider the wider context or category
        
        Guidelines:
        - Make queries specific and focused
        - Include model numbers and product names when present
        - Consider both exact matches and semantic variations
        - Avoid overly generic terms
        - Each query should be distinct and target different aspects
        
        Format: Return only the queries, one per line, no numbering or prefixes."""),
        ("human", question)
    ]
    
    response = llm.invoke(messages)
    queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    
    # Add the original question as one of the queries if not too similar
    if question not in queries and not any(similar(question, q) for q in queries):
        queries.append(question)
    
    # Debug logging
    print("\nGenerated search queries:")
    for q in queries:
        print(f"- {q}")
    
    return queries

def similar(str1: str, str2: str) -> bool:
    """Simple check if strings are very similar"""
    s1 = set(str1.lower().split())
    s2 = set(str2.lower().split())
    overlap = len(s1.intersection(s2))
    return overlap > min(len(s1), len(s2)) * 0.8

def rank_results(question: str, docs: List[Any]) -> List[Any]:
    if not docs:
        return []
    
    print(f"\nRanking {len(docs)} documents for question: {question}")
    
    messages = [
        ("system", """Analyze these document chunks and rank them based on their relevance to the user's question.
        Consider:
        - Direct answers vs indirect references
        - Specificity of information
        - Authoritative sources (e.g., dedicated feature pages over general mentions)
        - Completeness of information
        - Technical accuracy
        - Context requirements (whether the chunk needs additional context to be useful)
        
        For each document, assign a relevance score and explain why in one line.
        Format: <index>|<score>|<reason>
        Example: 2|0.95|Direct answer with technical specifications"""),
        ("human", f"Question: {question}\n\nDocuments:\n" + "\n\n".join([
            f"Document {i}:\nURL: {doc.metadata.get('url', 'N/A') if hasattr(doc, 'metadata') else 'N/A'}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]))
    ]
    
    response = llm.invoke(messages)
    try:
        # Parse scored results
        results = []
        for line in response.content.split('\n'):
            if '|' in line:
                idx, score, reason = line.split('|')
                print(f"Ranking - Document {idx.strip()}: score={score.strip()}, reason={reason.strip()}")
                results.append((int(idx.strip()), float(score.strip())))
        
        # Sort by score and return corresponding docs
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        ranked_docs = [docs[i] for i, _ in sorted_results if i < len(docs)]
        print(f"Ranked {len(ranked_docs)} documents")
        return ranked_docs
    except Exception as e:
        print(f"Error ranking results: {str(e)}")
        return docs  # Fall back to original order if ranking fails

def get_relevant_history(input_text: str) -> str:
    """Get relevant context by searching across all collections"""
    # Generate multiple search queries
    search_queries = generate_search_queries(input_text)
    
    # Search with each query
    all_results = []
    seen_urls = set()  # Track unique sources
    
    print("\nSearching with generated queries...")
    for query in search_queries:
        # Search across all available collections
        docs = collection_manager.search_collections(
            query=query,
            k=5  # Increased from 3 to get more potential matches
        )
        print(f"\nQuery '{query}' returned {len(docs)} results")
        
        # Deduplicate results while preserving order
        for doc in docs:
            if hasattr(doc, 'metadata'):
                url = doc.metadata.get('url', '')
                if url and url not in seen_urls:  # Only add if URL exists and is unique
                    seen_urls.add(url)
                    all_results.append(doc)
                    print(f"Added unique result from URL: {url}")
                    print(f"Content preview: {doc.page_content[:200]}...")
    
    # Analyze and rank results
    ranked_results = rank_results(input_text, all_results)
    
    # Format context with ranked results
    context_parts = []
    for doc in ranked_results:
        if hasattr(doc, 'metadata'):
            url = doc.metadata.get('url', '')
            title = doc.metadata.get('title', '') or "Documentation"
            rank_score = doc.metadata.get('rank_score', 0)
            rank_reason = doc.metadata.get('rank_reason', '')
            
            # Format source information
            if url:
                context_parts.append(
                    f"Source: {title}\n"
                    f"URL: {url}\n"
                    f"Relevance: {rank_score:.2f} - {rank_reason}\n"
                    f"Content: {doc.page_content}"
                )
            else:
                context_parts.append(
                    f"Source: {title}\n"
                    f"Relevance: {rank_score:.2f} - {rank_reason}\n"
                    f"Content: {doc.page_content}"
                )
    
    # Debug logging
    print(f"\nSearch queries generated: {search_queries}")
    print(f"Number of results found: {len(all_results)}")
    print(f"Number of ranked results: {len(ranked_results)}")
    if ranked_results:
        print("First result metadata:", 
              {k: v for k, v in ranked_results[0].metadata.items() 
               if k in ['url', 'title', 'rank_score', 'rank_reason']})
        print("First result content preview:", ranked_results[0].page_content[:200])
    
    return "\n\n".join(context_parts)

# Create Gradio interface
with gr.Blocks(
    title="HH Agent Tails",
    theme=gr.themes.Default(),
    analytics_enabled=False
) as interface:
    gr.Markdown("# HH Agent Tails")
    
    with gr.Tabs():
        with gr.Tab("Chat Interface"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Message")
            clear = gr.Button("Clear")

        with gr.Tab("Document Management"):
            if not is_admin_mode:
                gr.Markdown("""
                ### Read-Only Mode
                This instance is running in read-only mode. Document management features are disabled.
                
                To manage document collections:
                1. Clone the repository from GitHub
                2. Set up your own Supabase instance
                3. Run the application in admin mode with your Supabase credentials
                """)
            
            with gr.Group():
                gr.Markdown("### Available Collections")
                collections_table = gr.Dataframe(
                    headers=["Collection Name", "Document Count"],
                    datatype=["str", "number"],
                    label="Current Collections",
                    row_count=(1, "dynamic"),  # Start with 1 row, grow dynamically
                    height=100,  # Initial height
                    wrap=True  # Allow text wrapping
                )
            
            if is_admin_mode:
                with gr.Group():
                    gr.Markdown("### Collection Management")
                    with gr.Row():
                        collection_name_input = gr.Textbox(label="Collection Name")
                        create_collection_btn = gr.Button("Create Collection", variant="primary")
                    
                    with gr.Row():
                        collection_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Collection for Management",
                            interactive=True
                        )
                        clear_collection_btn = gr.Button("Clear Collection", variant="secondary")
                        delete_collection_btn = gr.Button("Delete Collection", variant="stop")
                    
                    collection_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Group():
                    gr.Markdown("### Document Upload")
                    with gr.Row():
                        upload_collection_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Collection for Upload",
                            interactive=True
                        )
                        file_upload = gr.File(file_count="multiple")
                    with gr.Row():
                        upload_btn = gr.Button("Upload Documents", variant="primary")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    gr.Markdown("""Note: Large files may take some time to process. 
                    Please wait for the upload status to update.""")

    # Function to update collections table
    def update_collections_table():
        collections = collection_manager.list_collections()
        rows = []
        for collection in collections:
            doc_count = collection_manager.get_collection(collection).count_documents()
            rows.append([collection, doc_count])
        return rows

    # Function to update dropdowns
    def update_dropdowns():
        collections = collection_manager.list_collections()
        return gr.Dropdown(choices=collections), gr.Dropdown(choices=collections)

    # Update collections table and dropdowns on page load
    interface.load(
        update_collections_table,
        outputs=[collections_table]
    )
    if is_admin_mode:
        interface.load(
            update_dropdowns,
            outputs=[collection_dropdown, upload_collection_dropdown]
        )

    if is_admin_mode:
        # Event handlers for admin mode
        create_collection_btn.click(
            create_collection_ui,
            inputs=[collection_name_input],
            outputs=[collection_status],
        ).then(
            update_collections_table,
            outputs=[collections_table]
        ).then(
            update_dropdowns,
            outputs=[collection_dropdown, upload_collection_dropdown]
        )
        
        clear_collection_btn.click(
            clear_collection_ui,
            inputs=[collection_dropdown],
            outputs=[collection_status]
        ).then(
            update_collections_table,
            outputs=[collections_table]
        )
        
        delete_collection_btn.click(
            delete_collection_ui,
            inputs=[collection_dropdown],
            outputs=[collection_status]
        ).then(
            update_collections_table,
            outputs=[collections_table]
        ).then(
            update_dropdowns,
            outputs=[collection_dropdown, upload_collection_dropdown]
        )
        
        upload_btn.click(
            upload_file,
            inputs=[file_upload, upload_collection_dropdown],
            outputs=[upload_status]
        ).then(
            update_collections_table,
            outputs=[collections_table]
        )
    
    msg.submit(process_chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")
