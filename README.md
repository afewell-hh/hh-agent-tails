# HH Agent Tails

[![Sync to Hugging Face hub](https://github.com/afewell-hh/hh-agent-tails/actions/workflows/deploy.yml/badge.svg)](https://github.com/afewell-hh/hh-agent-tails/actions/workflows/deploy.yml)

A powerful RAG (Retrieval-Augmented Generation) system built with FastAPI, Gradio, and ChromaDB. Try it on [Hugging Face Spaces](https://huggingface.co/spaces/afewell/hh-agent-tails)!

## Features

- **Chat Interface**: Interactive chat with context-aware responses and session tracking
- **Document Management**: Dynamic collection management supporting multiple document formats
- **GPT-Crawler Integration**: Use gpt-crawler for web scraping with accurate citations
- **Advanced RAG Implementation**: Multi-query retrieval, intelligent chunking, result ranking, and source attribution
- **Analytics and Logging**: Conversation logging to Supabase with detailed session tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/afewell-hh/hh-agent-tails.git
cd hh-agent-tails
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
uvicorn hh_agent_tails.app:app --reload
```

2. Open your browser and navigate to:
- Gradio Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Document Collections

1. Create a new collection through the interface
2. Upload documents (supports JSON, TXT)
3. Start chatting with context from your documents

## Deployment

### Local Development
1. Set environment variables in `.env`
2. Run with `uvicorn` as shown above

### Hugging Face Spaces
1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your GitHub repository
4. Add required secrets (OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details