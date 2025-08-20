import os
import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SIMILARITY_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
RESPONSE_MODE = "compact"  # Options: "compact", "tree_summarize", "no_text"

VECTOR_STORE_TYPE = "chroma"  # We're using ChromaDB
PERSIST_DIR = "./data/llama_index_storage"
CHROMA_COLLECTION = "llama_resumes"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = DATA_DIR / "llama_index_storage"
CHROMA_DIR = DATA_DIR / "chromadb_llama"

for directory in [DATA_DIR, STORAGE_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def initialize_llama_index(
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        debug: bool = False,
) -> None:
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model or EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        embed_batch_size=10,  # to avoid rate limits
    )

    Settings.llm = OpenAI(
        model=llm_model or LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1000,
    )

    Settings.text_splitter = SentenceSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        include_metadata=True,
        include_prev_next_rel=True,
    )

    Settings.context_window = 4096
    Settings.num_output = 512

    if debug:
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = callback_manager

    logger.info(
        f"LlamaIndex initialized with:\n"
        f"  - Embedding: {embedding_model or EMBEDDING_MODEL}\n"
        f"  - LLM: {llm_model or LLM_MODEL}\n"
        f"  - Chunk size: {chunk_size or CHUNK_SIZE}\n"
        f"  - Chunk overlap: {chunk_overlap or CHUNK_OVERLAP}"
    )


QA_PROMPT_TEMPLATE = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: 
"""

RESUME_ANALYSIS_PROMPT = """
You are an expert recruiter analyzing resumes.

Context (Resume Information):
---------------------
{context_str}
---------------------

Task: {query_str}

Provide a detailed analysis focusing on:
1. Relevant experience and skills
2. Years of experience
3. Key achievements
4. Suitability for the role

Answer:
"""

CANDIDATE_SELECTION_PROMPT = """
You are an expert technical recruiter tasked with selecting the best candidate.

Available Candidates:
---------------------
{context_str}
---------------------

Requirements: {query_str}

Select the best candidate and provide:
1. Selected candidate ID and name
2. Detailed reasoning for selection
3. Key strengths that match requirements
4. Any concerns or gaps
5. Ranking of all candidates

Respond in a structured format.
"""


class CostTracker:
    PRICING = {
        "text-embedding-3-small": 0.00002,  # per 1k tokens
        "text-embedding-3-large": 0.00013,
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.embedding_tokens = 0
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.total_cost = 0.0

    def track_embedding(self, num_tokens: int, model: str = EMBEDDING_MODEL):
        self.embedding_tokens += num_tokens
        if model in self.PRICING:
            cost = (num_tokens / 1000) * self.PRICING[model]
            self.total_cost += cost

    def track_llm(
            self,
            input_tokens: int,
            output_tokens: int,
            model: str = LLM_MODEL
    ):
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens

        if model in self.PRICING and isinstance(self.PRICING[model], dict):
            input_cost = (input_tokens / 1000) * self.PRICING[model]["input"]
            output_cost = (output_tokens / 1000) * self.PRICING[model]["output"]
            self.total_cost += input_cost + output_cost

    def get_summary(self) -> dict:
        return {
            "embedding_tokens": self.embedding_tokens,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "total_cost": round(self.total_cost, 4),
        }


cost_tracker = CostTracker()

initialize_llama_index()
