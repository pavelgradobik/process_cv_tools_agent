try:
    from backend.llama_config import (
        OPENAI_API_KEY,
        EMBEDDING_MODEL,
        LLM_MODEL,
        initialize_llama_index,
        cost_tracker,
    )
except ImportError:
    OPENAI_API_KEY = None
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    initialize_llama_index = None
    cost_tracker = None

try:
    from backend.llama_index_store import LlamaIndexStore, create_llama_store
except ImportError:
    LlamaIndexStore = None
    create_llama_store = None

try:
    from backend.llama_query_engine import (
        ResumeQueryEngine,
        SmartResumeAgent,
        QueryConfig,
    )
except ImportError:
    ResumeQueryEngine = None
    SmartResumeAgent = None
    QueryConfig = None

try:
    from backend.file_processor import ResumeProcessor, load_resumes
except ImportError:
    ResumeProcessor = None
    load_resumes = None

__all__ = [
    "OPENAI_API_KEY",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "initialize_llama_index",
    "cost_tracker",
    "LlamaIndexStore",
    "create_llama_store",
    "ResumeQueryEngine",
    "SmartResumeAgent",
    "QueryConfig",
    "ResumeProcessor",
    "load_resumes",
]