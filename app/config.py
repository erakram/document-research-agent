"""Settings, read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    model: str
    embedding_model: str
    db_path: Path
    request_timeout: float  # seconds per LLM request
    max_retries: int  # automatic retries on rate limits, timeouts and 5xx errors
    max_agent_steps: int  # max LLM calls per question
    max_upload_mb: float
    seed_sample_docs: bool  # load sample_docs/ on first start so the demo works immediately

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "minishlab/potion-base-8M"),
            db_path=Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "documents.db")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "5")),
            max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "10")),
            seed_sample_docs=os.getenv("SEED_SAMPLE_DOCS", "true").lower() == "true",
        )
