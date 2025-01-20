# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

# Hugging Face Spaces entry point
if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    print("\n=== Starting Application ===")
    print(f"Python version: {sys.version}")
    print(f"Environment variables present: {[k for k in os.environ.keys() if k in ['SPACE_ID', 'OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']]}")
    
    try:
        # Verify required environment variables
        required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        # Initialize core components
        print("\nInitializing components...")
        print("- Initializing LLM...")
        llm.invoke([{"role": "user", "content": "test"}])  # Test LLM initialization
        print("- LLM initialized successfully")
        
        print("- Testing Supabase connection...")
        if conversation_logger.supabase:
            # Just test the connection without querying
            conversation_logger.supabase.table('conversations').select("count").execute()
            print("- Supabase connection successful")
        
        print("\nStarting server on port 7860...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7860,
            log_level="info"
        )
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("\nTraceback:")
            traceback.print_tb(e.__traceback__)
        sys.exit(1)  # Exit with error code to make the issue more visible 