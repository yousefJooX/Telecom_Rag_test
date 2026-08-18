import os
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.config.config_parser import settings
from src.logging.logger import logger


class ModelFactory:
    """
    Factory Pattern + Singleton (@lru_cache) for instantiating ML models.
    Prevents re-loading heavy weights on every HTTP request.
    """

    @staticmethod
    @lru_cache(maxsize=1)
    def get_embeddings():
        if settings.embedding_provider.lower() == "huggingface":
            logger.info(f"Initializing Embeddings Model (Provider: {settings.embedding_provider})...")
            return HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True} # Normalize embeddings for better similarity search 
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    @staticmethod
    @lru_cache(maxsize=1)
    def get_llm():
        if settings.llm_provider.lower() == "gemini":
            logger.info(f"Initializing LLM (Provider: {settings.llm_provider})...")
            return ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                temperature=settings.temperature
            )
        elif settings.llm_provider.lower() == "openai":
            logger.info(f"Initializing LLM (Provider: {settings.llm_provider})...")
            return ChatOpenAI(
                model=settings.llm_model_name,
                temperature=settings.temperature
            )
        elif settings.llm_provider.lower() == "nvidia":
            logger.info(f"Initializing LLM (Provider: {settings.llm_provider})...")
            return ChatNVIDIA(
                model=settings.llm_model_name,
                temperature=settings.temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")