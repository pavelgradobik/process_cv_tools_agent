import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, FilterCondition, MetadataFilter

from backend.llama_config import (
    CHROMA_DIR,
    STORAGE_DIR,
    CHROMA_COLLECTION,
    SIMILARITY_TOP_K,
    RESPONSE_MODE,
    cost_tracker,
)

logger = logging.getLogger(__name__)


class LlamaIndexStore:
    def __init__(
            self,
            collection_name: str = CHROMA_COLLECTION,
            persist_dir: Optional[Path] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or CHROMA_DIR
        self.storage_dir = STORAGE_DIR

        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.index = None
        self.storage_context = None

        self.document_count = 0

        logger.info(f"Initialized LlamaIndexStore with collection: {collection_name}")

    def create_documents(
            self,
            resumes: List[Dict[str, Any]]
    ) -> List[Document]:
        documents = []

        for resume in resumes:
            metadata = resume.get("metadata", {})

            metadata.update({
                "resume_id": resume.get("id"),
                "source": "resume_database",
                "doc_type": "resume",
            })

            doc = Document(
                text=resume.get("text", ""),
                metadata=metadata,
                doc_id=resume.get("id"),
            )

            documents.append(doc)

        logger.info(f"Created {len(documents)} documents from resumes")
        return documents

    def create_nodes(
            self,
            documents: List[Document]
    ) -> List[TextNode]:
        from llama_index.core.node_parser import SentenceSplitter

        parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            include_metadata=True,
            include_prev_next_rel=True,
        )

        nodes = []
        for doc in documents:
            doc_nodes = parser.get_nodes_from_documents([doc])
            nodes.extend(doc_nodes)

        logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
        return nodes

    def build_index(
            self,
            resumes: List[Dict[str, Any]],
            rebuild: bool = False,
    ) -> VectorStoreIndex:
        if rebuild:
            self.reset_collection()

        documents = self.create_documents(resumes)

        chroma_collection = self.chroma_client.get_or_create_collection(
            self.collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        logger.info(f"Building index for {len(documents)} documents...")

        try:
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=True,
            )

            self.index.storage_context.persist(persist_dir=str(self.storage_dir))

            self.document_count = len(documents)

            total_tokens = sum(len(doc.text.split()) * 1.3 for doc in documents)
            cost_tracker.track_embedding(int(total_tokens))

            logger.info(f"Successfully indexed {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

        return self.index

    def load_index(self) -> Optional[VectorStoreIndex]:
        try:
            chroma_collection = self.chroma_client.get_collection(
                self.collection_name
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            self.storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(self.storage_dir),
            )

            self.index = load_index_from_storage(
                self.storage_context,
                index_id="vector_index",
            )

            self.document_count = len(chroma_collection.get()["ids"])

            logger.info(f"Loaded index with {self.document_count} documents")
            return self.index

        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            return None

    def create_query_engine(
            self,
            similarity_top_k: int = SIMILARITY_TOP_K,
            response_mode: str = RESPONSE_MODE,
            streaming: bool = False,
            filters: Optional[Dict[str, Any]] = None,
    ) -> RetrieverQueryEngine:
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            filters=filters,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            streaming=streaming,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ],
        )

        return query_engine

    def search(
            self,
            query: str,
            top_k: int = SIMILARITY_TOP_K,
            filters: Optional[Dict[str, Any]] = None,
            return_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")

        from llama_index.core.vector_stores import MetadataFilters, FilterCondition, MetadataFilter

        metadata_filters = None
        if filters:
            filter_list = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    if "$gte" in value:
                        filter_list.append(
                            MetadataFilter(
                                key=key,
                                value=value["$gte"],
                                operator=">="
                            )
                        )
                    elif "$lte" in value:
                        filter_list.append(
                            MetadataFilter(
                                key=key,
                                value=value["$lte"],
                                operator="<="
                            )
                        )
                    elif "$eq" in value:
                        filter_list.append(
                            MetadataFilter(
                                key=key,
                                value=value["$eq"],
                                operator="=="
                            )
                        )
                else:
                    # Simple equality filter
                    filter_list.append(
                        MetadataFilter(
                            key=key,
                            value=value,
                            operator="=="
                        )
                    )

            if filter_list:
                metadata_filters = MetadataFilters(
                    filters=filter_list,
                    condition=FilterCondition.AND
                )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=metadata_filters,
        )

        nodes = retriever.retrieve(query)

        results = []
        for node_with_score in nodes:
            node = node_with_score.node

            result = {
                "id": node.node_id,
                "text": node.text,
                "score": node_with_score.score,
            }

            if return_metadata:
                result["metadata"] = node.metadata

            results.append(result)

        query_tokens = len(query.split()) * 1.3
        cost_tracker.track_embedding(int(query_tokens))

        return results

    def query(
            self,
            query: str,
            response_mode: str = "compact",
            streaming: bool = False,
    ) -> str:
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")

        query_engine = self.create_query_engine(
            response_mode=response_mode,
            streaming=streaming,
        )

        response = query_engine.query(query)

        input_tokens = len(query.split()) * 1.3
        output_tokens = len(str(response).split()) * 1.3
        cost_tracker.track_llm(int(input_tokens), int(output_tokens))

        return str(response)

    def analyze_candidate(
            self,
            candidate_id: str,
            requirements: str,
    ) -> Dict[str, Any]:
        query_engine = self.create_query_engine(
            filters={"resume_id": candidate_id},
            response_mode="tree_summarize",
        )

        query = f"""
        Analyze this candidate against the following requirements:
        {requirements}

        Provide:
        1. Match score (0-10)
        2. Matching skills
        3. Missing skills
        4. Overall assessment
        """

        response = query_engine.query(query)

        return {
            "candidate_id": candidate_id,
            "analysis": str(response),
        }

    def batch_analyze(
            self,
            candidate_ids: List[str],
            requirements: str,
    ) -> List[Dict[str, Any]]:
        results = []

        for candidate_id in candidate_ids:
            try:
                analysis = self.analyze_candidate(candidate_id, requirements)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing candidate {candidate_id}: {e}")
                results.append({
                    "candidate_id": candidate_id,
                    "error": str(e),
                })

        return results

    def reset_collection(self):
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception:
            pass

        self.chroma_client.create_collection(self.collection_name)
        logger.info(f"Created new collection: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "collection_name": self.collection_name,
            "document_count": self.document_count,
            "index_built": self.index is not None,
        }

        if self.index:
            try:
                collection = self.chroma_client.get_collection(self.collection_name)
                stats["total_chunks"] = collection.count()
            except:
                pass

        stats["cost_summary"] = cost_tracker.get_summary()

        return stats


def create_llama_store(**kwargs) -> LlamaIndexStore:
    return LlamaIndexStore(**kwargs)


def get_default_store() -> LlamaIndexStore:
    return LlamaIndexStore()
