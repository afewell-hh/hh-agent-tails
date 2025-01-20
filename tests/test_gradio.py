import pytest
import tempfile
import json
import os
from pathlib import Path
from hh_agent_tails.app import create_interface, collection_manager
import gradio as gr

@pytest.fixture
def interface():
    return create_interface()

@pytest.fixture(autouse=True)
def cleanup_collections():
    """Clean up collections before and after each test"""
    # Clean up before test
    for name in list(collection_manager.collections.keys()):
        try:
            collection_manager.delete_collection(name)
        except Exception as e:
            print(f"Error cleaning up collection {name}: {e}")
    yield
    # Clean up after test
    for name in list(collection_manager.collections.keys()):
        try:
            collection_manager.delete_collection(name)
        except Exception as e:
            print(f"Error cleaning up collection {name}: {e}")

def test_create_collection(interface):
    create_fn = interface.fns[0]
    result = create_fn.fn("test_collection")
    assert "Created new collection" in result[0]
    assert isinstance(result[1], list)  # Collection status should be a list
    assert "choices" in result[2]  # Dropdown choices should be a dict with choices
    assert "choices" in result[3]  # Dropdown choices should be a dict with choices

def test_upload_document(interface):
    # First create a collection
    create_fn = interface.fns[0]
    create_fn.fn("test_collection")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
        json.dump([{
            "url": "https://docs.example.com/test",
            "title": "Test Document",
            "html": "Test content",
            "links": ["https://docs.example.com/test"]
        }], temp_file)
        temp_file.flush()
        try:
            upload_fn = interface.fns[1]
            result = upload_fn.fn([temp_file.name], "test_collection")
            assert "Successfully uploaded" in result[0]
            assert isinstance(result[1], list)  # Collection status should be a list
        finally:
            os.unlink(temp_file.name)

def test_chat_interface(interface):
    """Test chat functionality"""
    # Create a mock message and history
    message = "Hello"
    history = []
    
    # Find the chat components
    chatbot = None
    msg = None
    for component in interface.blocks.values():
        if isinstance(component, gr.Chatbot):
            chatbot = component
        elif isinstance(component, gr.Textbox) and component.label == "Message":
            msg = component
    
    assert chatbot is not None, "Could not find chatbot component"
    assert msg is not None, "Could not find message textbox component"
    
    # Test chat functionality by directly calling predict
    from hh_agent_tails.app import predict
    response = predict(message, history)
    
    # Check the response
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    
    # Test the respond function
    def respond(message, chat_history):
        bot_message = predict(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    result = respond(message, history)
    
    # Check the results
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should have two elements"
    assert isinstance(result[0], str), "First element should be a string"
    assert isinstance(result[1], list), "Second element should be a list"
    assert len(result[1]) > 0, "Chat history should not be empty"
    assert isinstance(result[1][0], tuple), "Chat history entries should be tuples"
    assert len(result[1][0]) == 2, "Chat history entries should have two elements"
    assert result[1][0][0] == message, "First message in history should match input"

def test_clear_collection(interface):
    # First create a collection and add a document
    create_fn = interface.fns[0]
    create_fn.fn("test_collection")
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
        json.dump([{
            "url": "https://docs.example.com/test",
            "title": "Test Document",
            "html": "Test content",
            "links": ["https://docs.example.com/test"]
        }], temp_file)
        temp_file.flush()
        try:
            upload_fn = interface.fns[1]
            upload_fn.fn([temp_file.name], "test_collection")
            
            clear_fn = interface.fns[3]
            result = clear_fn.fn("test_collection", "clear")
            assert "Cleared collection" in result[0]
            assert isinstance(result[1], list)  # Collection status should be a list
            assert "choices" in result[2]  # Dropdown choices should be a dict with choices
            assert "choices" in result[3]  # Dropdown choices should be a dict with choices
        finally:
            os.unlink(temp_file.name)

def test_delete_collection(interface):
    # First create a collection
    create_fn = interface.fns[0]
    create_fn.fn("test_collection")
    
    delete_fn = interface.fns[3]  # Delete function is at index 3
    result = delete_fn.fn("test_collection", "delete")
    assert "Deleted collection" in result[0]
    assert isinstance(result[1], list)  # Collection status should be a list
    assert "choices" in result[2]  # Dropdown choices should be a dict with choices
    assert "choices" in result[3]  # Dropdown choices should be a dict with choices

def test_collection_status_update(interface):
    # First create a collection
    create_fn = interface.fns[0]
    result = create_fn.fn("test_collection")
    assert isinstance(result[1], list)  # Collection status should be a list

def test_error_handling(interface):
    # Test duplicate collection creation
    create_fn = interface.fns[0]
    create_fn.fn("test_collection")
    result = create_fn.fn("test_collection")
    assert "Error creating collection" in result[0] 