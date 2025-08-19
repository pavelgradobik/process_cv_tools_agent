import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chromadb"

for directory in [DATA_DIR, UPLOADS_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = DATA_DIR / "Resume.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please add it to your .env file."
    )

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

MODEL_PRICING = {
    "text-embedding-3-small": {"input": 0.02},
    "text-embedding-3-large": {"input": 0.13},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resume_embeddings")

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))

MAX_TOKENS_PER_REQUEST = 8191
MAX_CONTEXT_LENGTH = 4096

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.5"))

SEARCH_WEIGHTS = {
    "semantic_similarity": 0.7,
    "keyword_match": 0.2,
    "metadata_match": 0.1,
}

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = 1

REQUESTS_PER_MINUTE = 500
TOKENS_PER_MINUTE = 200000

ROLE_SYNONYMS: Dict[str, List[str]] = {
    "software engineer": [
        "software developer", "swe", "software engineer", "programmer",
        "developer", "coder", "software architect"
    ],
    "data scientist": [
        "data scientist", "ml engineer", "machine learning engineer",
        "ai engineer", "data analyst", "research scientist"
    ],
    "product manager": [
        "product manager", "pm", "product owner", "po",
        "product lead", "program manager"
    ],
    "designer": [
        "designer", "ux designer", "ui designer", "product designer",
        "ux/ui designer", "user experience designer", "visual designer"
    ],
    "hr": [
        "hr", "human resources", "recruiter", "talent acquisition",
        "people operations", "hr manager", "talent partner"
    ],
    "qa": [
        "qa", "quality assurance", "test engineer", "sdet",
        "qa engineer", "tester", "automation engineer"
    ],
    "devops": [
        "devops", "site reliability engineer", "sre", "cloud engineer",
        "infrastructure engineer", "platform engineer"
    ],
    "frontend": [
        "frontend", "front-end", "frontend developer", "ui developer",
        "react developer", "angular developer", "vue developer"
    ],
    "backend": [
        "backend", "back-end", "backend developer", "server developer",
        "api developer", "microservices developer"
    ],
    "fullstack": [
        "fullstack", "full-stack", "full stack developer",
        "generalist engineer", "web developer"
    ],
}

SKILL_CATEGORIES = {
    "programming": ["python", "java", "javascript", "c++", "go", "rust", "typescript"],
    "frontend": ["react", "angular", "vue", "html", "css", "sass", "webpack"],
    "backend": ["django", "flask", "spring", "express", "fastapi", "rails"],
    "database": ["sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"],
    "cloud": ["aws", "gcp", "azure", "kubernetes", "docker", "terraform"],
    "data": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "spark"],
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def validate_config():
    """Validate configuration settings."""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")

    if CHUNK_SIZE < 100:
        errors.append(f"CHUNK_SIZE ({CHUNK_SIZE}) is too small, minimum 100")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})")

    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

validate_config()