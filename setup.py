from setuptools import setup, find_packages

setup(
    name="hh-agent-tails",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "gradio",
        "langchain",
        "langchain-openai",
        "langchain-community",
        "langchain-core",
        "python-dotenv",
        "uvicorn",
    ],
) 