# Create the interface
interface = create_interface()

# Mount Gradio app to FastAPI
app = FastAPI()
app = gr.mount_gradio_app(app, interface, path="/")

# For Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hh_agent_tails.app:app",  # Use module path format
        host="0.0.0.0",
        port=7860,
        reload=False
    ) 