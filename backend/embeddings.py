import time
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from backend.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    MAX_TOKENS_PER_REQUEST,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    MODEL_PRICING,
)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    def __init__(
            self,
            model: str = OPENAI_EMBEDDING_MODEL,
            api_key: Optional[str] = None,
            batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or OPENAI_API_KEY

        self.client = OpenAI(api_key=self.api_key, timeout=REQUEST_TIMEOUT)

        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Using fallback tokenizer for model {model}")

        self.total_tokens_used = 0
        self.total_cost = 0.0

        logger.info(f"Initialized OpenAIEmbedder with model: {model}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def validate_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        cleaned_texts = []
        skipped_indices = []

        for i, text in enumerate(texts):
            text = text.strip()

            if not text:
                logger.warning(f"Skipping empty text at index {i}")
                skipped_indices.append(i)
                continue

            token_count = self.count_tokens(text)
            if token_count > MAX_TOKENS_PER_REQUEST:
                logger.warning(
                    f"Text at index {i} exceeds token limit "
                    f"({token_count} > {MAX_TOKENS_PER_REQUEST}). Truncating..."
                )
                tokens = self.tokenizer.encode(text)[:MAX_TOKENS_PER_REQUEST]
                text = self.tokenizer.decode(tokens)

            cleaned_texts.append(text)

        return cleaned_texts, skipped_indices

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            if hasattr(response, 'usage'):
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used

                if self.model in MODEL_PRICING:
                    cost = (tokens_used / 1_000_000) * MODEL_PRICING[self.model]["input"]
                    self.total_cost += cost

            return embeddings

        except Exception as e:
            logger.error(f"Error in embedding batch: {e}")
            raise

    def embed_texts(
            self,
            texts: List[str],
            show_progress: bool = True,
    ) -> np.ndarray:

        if not texts:
            return np.array([])

        cleaned_texts, skipped_indices = self.validate_texts(texts)

        if not cleaned_texts:
            logger.warning("No valid texts to embed after validation")
            return np.array([])

        embeddings = []
        total_batches = (len(cleaned_texts) + self.batch_size - 1) // self.batch_size

        logger.info(f"Embedding {len(cleaned_texts)} texts in {total_batches} batches")

        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches}")

            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)

                if i + self.batch_size < len(cleaned_texts):
                    time.sleep(0.1)  # Small delay between batches

            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {e}")
                if embeddings:
                    logger.warning("Returning partial embeddings")
                    break
                raise

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        logger.info(
            f"Successfully embedded {len(embeddings)} texts. "
            f"Total tokens: {self.total_tokens_used:,}, "
            f"Total cost: ${self.total_cost:.4f}"
        )

        return embeddings_array

    def embed_single(self, text: str) -> np.ndarray:
        embeddings = self.embed_texts([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def get_embedding_dimension(self) -> int:
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)

    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        total_tokens = sum(self.count_tokens(text) for text in texts)

        if self.model in MODEL_PRICING:
            cost = (total_tokens / 1_000_000) * MODEL_PRICING[self.model]["input"]
        else:
            cost = 0.0

        return {
            "total_texts": len(texts),
            "total_tokens": total_tokens,
            "estimated_cost": cost,
            "model": self.model,
        }

    def reset_usage_stats(self):
        self.total_tokens_used = 0
        self.total_cost = 0.0
        logger.info("Usage statistics reset")

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "cost_per_1k_tokens": MODEL_PRICING.get(self.model, {}).get("input", 0) / 1000,
        }

def create_embedder(**kwargs) -> OpenAIEmbedder:
    return OpenAIEmbedder(**kwargs)