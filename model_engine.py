import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'openai')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_llm(model_provider=None, model_name=None):
    provider = model_provider or MODEL_PROVIDER
    model = model_name or MODEL_NAME

    if provider.lower() == 'openai':
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")
        return ChatOpenAI(model=model, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    elif provider.lower() == 'ollama':
        return ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0.2)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
