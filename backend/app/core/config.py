"""
Application configuration settings.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Telco Customer Churn Dashboard"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Supabase Configuration
    SUPABASE_URL: str = Field(env="SUPABASE_URL")
    SUPABASE_SERVICE_KEY: str = Field(env="SUPABASE_SERVICE_KEY")
    SUPABASE_ANON_KEY: str = Field(env="SUPABASE_ANON_KEY")
    
    # Database Configuration (fallback PostgreSQL)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis Configuration (for Celery)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # ML Model Configuration
    MODEL_PATH: str = Field(default="./data/ml_models", env="MODEL_PATH")
    MODEL_RETRAIN_THRESHOLD: float = Field(default=0.05, env="MODEL_RETRAIN_THRESHOLD")
    
    # RAG Configuration
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    QDRANT_COLLECTION_NAME: str = Field(default="telco-documents", env="QDRANT_COLLECTION_NAME")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="qwen3:1.7b", env="OLLAMA_MODEL")
    OLLAMA_EMBEDDING_MODEL: str = Field(default="snowflake-arctic-embed", env="OLLAMA_EMBEDDING_MODEL")
    
    # DeepSeek Chat Configuration
    DEEPSEEK_MODEL: str = Field(default="deepseek-r1:8b", env="DEEPSEEK_MODEL")
    DEEPSEEK_TEMPERATURE: float = Field(default=0.6, env="DEEPSEEK_TEMPERATURE")
    DEEPSEEK_MAX_TOKENS: int = Field(default=4096, env="DEEPSEEK_MAX_TOKENS")
    
    # File Upload Configuration
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".txt", ".docx", ".md"]
    
    # Security Configuration
    SECRET_KEY: str = Field(env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get settings instance."""
    return settings