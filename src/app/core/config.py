from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", alias="GROQ_MODEL")

    llm_provider: str = Field("gemini", alias="LLM_PROVIDER")
    llm_model: str = Field("gemini-2.5-flash", alias="LLM_MODEL")
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")

    # OpenAI-compatible providers (Lightning AI, OpenAI itself, vLLM, etc.)
    llm_base_url: str = Field("", alias="LLM_BASE_URL")
    llm_api_key: str = Field("", alias="LLM_API_KEY")

    embedding_model: str = Field("intfloat/multilingual-e5-large", alias="EMBEDDING_MODEL")
    reranker_model: str = Field("BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_device: str = Field("cuda", alias="RERANKER_DEVICE")
    reranker_enabled: bool = Field(True, alias="RERANKER_ENABLED")

    data_dir: Path = Field(Path("data"), alias="DATA_DIR")
    artifacts_dir: Path = Field(Path("artifacts"), alias="ARTIFACTS_DIR")

    dense_top_k: int = Field(20, alias="DENSE_TOP_K")
    bm25_top_k: int = Field(20, alias="BM25_TOP_K")
    rrf_k: int = Field(60, alias="RRF_K")
    rerank_top_k: int = Field(5, alias="RERANK_TOP_K")
    min_score: float = Field(0.0, alias="MIN_SCORE")

    chunk_max_chars: int = Field(700, alias="CHUNK_MAX_CHARS")

    n8n_webhook_url: str = Field(..., alias="N8N_WEBHOOK_URL")
    n8n_timeout_seconds: int = Field(5, alias="N8N_TIMEOUT_SECONDS")

    session_max_turns: int = Field(3, alias="SESSION_MAX_TURNS")

    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def data_path(self) -> Path:
        return (PROJECT_ROOT / self.data_dir).resolve()

    @property
    def artifacts_path(self) -> Path:
        return (PROJECT_ROOT / self.artifacts_dir).resolve()


settings = Settings()
