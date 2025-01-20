# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

# Hugging Face Spaces entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860  # Default Hugging Face port
    ) 