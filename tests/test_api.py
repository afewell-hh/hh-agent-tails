import pytest
from fastapi.testclient import TestClient
import json
import os
from hh_agent_tails.app import app, collection_manager

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_collection_name():
    return "test_collection"

@pytest.fixture
def sample_doc_json(tmp_path):
    """Create a sample JSON document for testing"""
    doc = {
        "content": "This is a test document for the RAG system.",
        "metadata": {
            "source": "test",
            "timestamp": "2024-01-19T00:00:00Z"
        }
    }
    file_path = tmp_path / "test_doc.json"
    with open(file_path, "w") as f:
        json.dump(doc, f)
    return str(file_path)

def test_create_collection(client, test_collection_name):
    """Test creating a new collection"""
    response = client.post("/create_collection", json={"name": test_collection_name})
    assert response.status_code == 200
    assert test_collection_name in collection_manager.collections

def test_create_duplicate_collection(client, test_collection_name):
    """Test creating a collection that already exists"""
    # First creation
    client.post("/create_collection", json={"name": test_collection_name})
    # Second creation should fail
    response = client.post("/create_collection", json={"name": test_collection_name})
    assert response.status_code == 409

def test_document_upload(client, test_collection_name, sample_doc_json):
    """Test uploading a document to a collection"""
    # Create collection
    client.post("/create_collection", json={"name": test_collection_name})
    
    # Upload document
    with open(sample_doc_json, "rb") as f:
        response = client.post(
            f"/upload_document/{test_collection_name}",
            files={"file": ("test_doc.json", f, "application/json")}
        )
    assert response.status_code == 200
    
    # Verify document was added
    collection = collection_manager.get_collection(test_collection_name)
    result = collection.client.table("documents").select("*").eq(
        "collection_name", test_collection_name
    ).execute()
    assert len(result.data) > 0

def test_collection_clear(client, test_collection_name, sample_doc_json):
    """Test clearing a collection"""
    # Create and populate collection
    client.post("/create_collection", json={"name": test_collection_name})
    with open(sample_doc_json, "rb") as f:
        client.post(
            f"/upload_document/{test_collection_name}",
            files={"file": ("test_doc.json", f, "application/json")}
        )
    
    # Clear collection
    response = client.post(f"/clear_collection/{test_collection_name}")
    assert response.status_code == 200
    
    # Verify collection is empty
    collection = collection_manager.get_collection(test_collection_name)
    result = collection.client.table("documents").select("*").eq(
        "collection_name", test_collection_name
    ).execute()
    assert len(result.data) == 0

def test_collection_delete(client, test_collection_name):
    """Test deleting a collection"""
    # Create collection
    client.post("/create_collection", json={"name": test_collection_name})
    
    # Delete collection
    response = client.delete(f"/delete_collection/{test_collection_name}")
    assert response.status_code == 200
    assert test_collection_name not in collection_manager.collections

def test_collection_persistence(client, test_collection_name, sample_doc_json):
    """Test that collections persist between server restarts"""
    # Create and populate collection
    client.post("/create_collection", json={"name": test_collection_name})
    with open(sample_doc_json, "rb") as f:
        client.post(
            f"/upload_document/{test_collection_name}",
            files={"file": ("test_doc.json", f, "application/json")}
        )
    
    # Simulate server restart by reinitializing collection manager
    collection_manager.__init__()
    
    # Verify collection still exists and contains documents
    assert test_collection_name in collection_manager.collections
    collection = collection_manager.get_collection(test_collection_name)
    result = collection.client.table("documents").select("*").eq(
        "collection_name", test_collection_name
    ).execute()
    assert len(result.data) > 0 