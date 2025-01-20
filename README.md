---
title: HH Agent Tails
emoji: 🦔
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.14.0"
app_file: hh_agent_tails/app.py
pinned: false
---

# HH Agent Tails - Hedgehog Documentation Assistant

HH Agent Tails is an AI-powered documentation assistant for the Hedgehog Open Network Fabric project. It uses advanced RAG (Retrieval-Augmented Generation) techniques to provide accurate answers about Hedgehog's features, capabilities, and supported devices.

## Features

- **Smart Document Search**: Uses vector similarity search with Supabase pgvector to find relevant documentation
- **Context-Aware Responses**: Provides answers based on official Hedgehog documentation with source citations
- **User-Friendly Interface**: Simple chat interface built with Gradio
- **Production Ready**: Deployed on Hugging Face Spaces with read-only mode for safe public access

## Technical Details

- **Vector Store**: Supabase with pgvector extension for efficient similarity search
- **Embeddings**: OpenAI embeddings for document vectorization
- **LLM**: GPT-3.5-turbo for natural language understanding and response generation
- **Document Processing**: Handles JSON documentation with 1500-token chunks and 300-token overlap
- **Backend**: FastAPI for robust API endpoints
- **Frontend**: Gradio for the chat interface

## Usage

Simply type your questions about Hedgehog in the chat interface. The assistant will search through the documentation and provide relevant answers with citations to source documents.

Example questions:
- "What leaf switches does Hedgehog support?"
- "Does Hedgehog support the Edgecore DCS203?"
- "How does Hedgehog handle multi-tenancy?"

## Development

To run locally in development mode:
```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run in development mode
ADMIN_MODE=true APP_ENV=development .venv/bin/python -m uvicorn hh_agent_tails.app:app --reload
```

## Deployment

The app automatically deploys to Hugging Face Spaces when changes are pushed to the main branch. In production, the app runs in read-only mode for security.

## License

This project is part of the Hedgehog Open Network Fabric ecosystem and follows its licensing terms.