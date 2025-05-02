from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Notion API設定
    notion_token: str
    notion_page_id: Optional[str] = None  # 親ページID
    
    # LLM設定
    llm_model: str = "qwen3:4b"  # より軽量なgemma:2bモデルを使用
    ollama_api_base: str = "http://host.docker.internal:11434/api"  # ホストマシンのOllamaにアクセス
    
    # 埋め込みモデル設定
    embedding_model: str = "intfloat/multilingual-e5-small"
    
    # ベクトルストア設定
    vector_store_path: str = "data"
    
    # RAG設定
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()