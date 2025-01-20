import pytest
import os
from hh_agent_tails.app import app, collection_manager
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up test environment variables and directories"""
    # Store original environment variables
    original_env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "dummy-key"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY")
    }

    # Set environment variables for testing
    os.environ["OPENAI_API_KEY"] = "dummy-key"
    os.environ["SUPABASE_URL"] = "http://localhost:54321"  # Local Supabase instance
    os.environ["SUPABASE_KEY"] = "dummy-key"

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

@pytest.fixture(autouse=True)
def setup_collection_manager():
    """Reset collection manager before each test"""
    # Reinitialize the collection manager
    collection_manager.__init__()

    yield collection_manager

    # Cleanup after test
    for name in list(collection_manager.collections.keys()):
        try:
            collection_manager.delete_collection(name)
        except Exception as e:
            print(f"Error cleaning up collection {name}: {e}")

    collection_manager.collections.clear()

@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)

@pytest.fixture
def test_collection_name():
    """Return a test collection name"""
    return "test_collection"

@pytest.fixture
def sample_doc_json(setup_test_environment):
    """Create a sample JSON document for testing"""
    doc_path = os.path.join(setup_test_environment, "test_doc.json")
    doc_content = {
        "url": "https://docs.example.com/test",
        "content": "The EdgeCore AS7326-56X is a high-performance leaf switch.",
        "metadata": {
            "source": "https://docs.example.com/test",
            "title": "Switch Documentation"
        }
    }
    
    with open(doc_path, "w") as f:
        json.dump(doc_content, f)
    
    return doc_path 