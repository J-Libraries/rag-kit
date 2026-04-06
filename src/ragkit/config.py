from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RAGConfig:
    persist_directory: str | Path = "db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    use_multi_query: bool = True
    enable_query_rewrite: bool = True
    collection_name: str = "default"
    verbose: bool = False

    llm_provider: str = "sarvam"
    llm_model: str | None = None
    llm_temperature: float = 0.2
    llm_kwargs: dict[str, Any] = field(default_factory=dict)