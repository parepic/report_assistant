import os
from functools import lru_cache
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from dotenv import load_dotenv

from app.data_classes import GlobalConfig
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.utils.load_utils import load_global_config


@lru_cache(maxsize=1)
def get_config() -> GlobalConfig:
    load_dotenv()
    return load_global_config()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClientWrapper:
    load_dotenv()
    config = get_config()
    return QdrantClientWrapper(config)


@lru_cache(maxsize=1)
def get_openai_client_chatbot() -> OpenAIClientWrapper:
    load_dotenv()
    config = get_config()
    return OpenAIClientWrapper(
        api_key=os.getenv("OPENAI_API_KEY"),
        llm_model=config.LLM_MODEL_CHATBOT,
    )


@lru_cache(maxsize=1)
def get_openai_client_comparison() -> OpenAIClientWrapper:
    load_dotenv()
    config = get_config()
    return OpenAIClientWrapper(
        api_key=os.getenv("OPENAI_API_KEY"),
        llm_model=config.LLM_MODEL_SUMMARIZER,
    )

@lru_cache(maxsize=1)
def get_engine():
    config = get_config()
    return create_async_engine(config.POSTGRESQL_URL, pool_pre_ping=True)

@lru_cache(maxsize=1)
def get_session_factory():
    return async_sessionmaker(get_engine(), expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    SessionLocal = get_session_factory()
    async with SessionLocal() as session:
        yield session
