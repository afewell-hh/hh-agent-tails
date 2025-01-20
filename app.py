# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

# Create the interface
demo = create_interface()

# For Hugging Face Spaces, we need to expose the demo interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 