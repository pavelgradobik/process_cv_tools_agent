import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from backend.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
    SEARCH_WEIGHTS,
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
            self,
            persist_dir: Optional[str] = None,
            collection_name: Optional[str] = None,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or COLLECTION_NAME

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )

        self.collection = self._get_or_create_collection()

        logger.info(
            f"Initialized VectorStore with collection '{self.collection_name}' "
            f"at {self.persist_dir}"
        )

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Created fresh collection: {self.collection_name}")

    def add_documents(
            self,
            documents: List[Dict[str, Any]],
            batch_size: int = 100,
    ) -> int:
        if not documents:
            return 0

        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for doc in batch:
                if not doc.get("id") or not doc.get("text"):
                    logger.warning(f"Skipping invalid document: {doc}")
                    continue

                ids.append(str(doc["id"]))
                texts.append(doc["text"])

                metadata = doc.get("metadata", {})

                clean_metadata = self._clean_metadata(metadata)
                metadatas.append(clean_metadata)

            if ids:
                try:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                    )
                    total_added += len(ids)
                    logger.info(f"Added batch {i // batch_size + 1}: {len(ids)} documents")
                except Exception as e:
                    logger.error(f"Error adding batch: {e}")

        logger.info(f"Total documents added: {total_added}")
        return total_added

    def search(
            self,
            query: str,
            top_k: int = DEFAULT_TOP_K,
            filter_metadata: Optional[Dict[str, Any]] = None,
            min_score: float = MIN_SIMILARITY_SCORE,
    ) -> List[Dict[str, Any]]:
        where_clause = self._build_where_clause(filter_metadata)

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results and results["ids"]:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                similarity = 1.0 - distances[i]

                if similarity >= min_score:
                    search_results.append({
                        "id": ids[i],
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "similarity": round(similarity, 4),
                        "distance": distances[i],
                    })

        return search_results

    def hybrid_search(
            self,
            query: str,
            keywords: List[str] = None,
            top_k: int = DEFAULT_TOP_K,
            filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        semantic_results = self.search(
            query=query,
            top_k=top_k * 2,
            filter_metadata=filter_metadata,
            min_score=0,
        )

        if keywords:
            for result in semantic_results:
                keyword_score = self._calculate_keyword_score(
                    result["text"],
                    keywords
                )

                result["keyword_score"] = keyword_score
                result["combined_score"] = (
                        SEARCH_WEIGHTS["semantic_similarity"] * result["similarity"] +
                        SEARCH_WEIGHTS["keyword_match"] * keyword_score
                )
        else:
            for result in semantic_results:
                result["combined_score"] = result["similarity"]

        semantic_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return semantic_results[:top_k]

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")

        return None

    def update_document(
            self,
            doc_id: str,
            text: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            existing = self.get_by_id(doc_id)
            if not existing:
                logger.warning(f"Document {doc_id} not found")
                return False

            new_text = text if text is not None else existing["text"]
            new_metadata = metadata if metadata is not None else existing["metadata"]

            self.collection.update(
                ids=[doc_id],
                documents=[new_text],
                metadatas=[self._clean_metadata(new_metadata)],
            )

            logger.info(f"Updated document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents(self, doc_ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_dir),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
        }

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}

        for key, value in metadata.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                clean[key] = json.dumps(value)
            else:
                clean[key] = str(value)

        return clean

    def _build_where_clause(
            self,
            filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not filter_metadata:
            return None

        where_clause = {}

        for key, value in filter_metadata.items():
            if isinstance(value, dict):
                where_clause[key] = value
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value

        return where_clause if where_clause else None

    def _calculate_keyword_score(
            self,
            text: str,
            keywords: List[str]
    ) -> float:
        if not keywords:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)

        return min(1.0, matches / len(keywords))

def create_vector_store(**kwargs) -> VectorStore:
    return VectorStore(**kwargs)


def get_default_store() -> VectorStore:
    return VectorStore()
