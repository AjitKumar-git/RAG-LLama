import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from ..models.adapter import BaseLlamaAdapter, OllamaLlamaAdapter, create_llama_adapter
from ..models.context_manager import LongContextManager, ContextWindowConfig
from ..network.context_network import ContextNetwork, ContextNode
from .multi_context import MultiContextManager, MultiContextDomain


def extract_keywords(query: str) -> List[str]:
    query_clean = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
    stop_words = {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "with", "on", "for", "at", "by", "from",
        "what", "where", "who", "which", "how", "why", "is", "it", "this", "that", "these", "those"
    }
    return [w for w in query_clean.split() if w not in stop_words and len(w) > 1]


class ReciprocalRankFusion:
    """Computes Reciprocal Rank Fusion (RRF) scores over multiple candidate result rankings."""

    @staticmethod
    def fuse_rankings(
        rankings: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines multiple ranked lists using RRF formula:
        RRF_score(doc) = sum(1 / (k + rank))
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank_list in rankings:
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc["id"]
                doc_map[doc_id] = doc
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_doc_ids = sorted(scores.keys(), key=lambda d_id: scores[d_id], reverse=True)
        fused_results = []
        for d_id in sorted_doc_ids:
            doc_item = dict(doc_map[d_id])
            doc_item["rrf_score"] = round(scores[d_id], 5)
            fused_results.append(doc_item)

        return fused_results


class SuperiorRAG:
    """
    Superior RAG Engine with Hybrid Search (dense vector + sparse BM25/keyword),
    Reciprocal Rank Fusion, Context Network graph expansion, Multi-Context management,
    and long-context model execution.
    """

    def __init__(
        self,
        llama_adapter: Optional[BaseLlamaAdapter] = None,
        multi_context_manager: Optional[MultiContextManager] = None,
        context_manager: Optional[LongContextManager] = None,
        embedding_model_name: str = "plutonioumguy/bge-m3"
    ):
        self.adapter = llama_adapter or OllamaLlamaAdapter()
        self.context_manager = context_manager or LongContextManager(
            ContextWindowConfig(max_context_tokens=self.adapter.get_max_context_length())
        )
        self.multi_context_manager = multi_context_manager or MultiContextManager()
        self.embedding_model_name = embedding_model_name

    def keyword_sparse_search(self, query: str, docs: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Sparse BM25/Keyword frequency search."""
        keywords = extract_keywords(query)
        if not keywords or not docs:
            return docs[:top_k]

        scored_docs = []
        for doc in docs:
            text_lower = doc["text"].lower()
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                doc_copy = dict(doc)
                doc_copy["sparse_score"] = score
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d.get("sparse_score", 0), reverse=True)
        return scored_docs[:top_k]

    def dense_vector_search(self, query: str, docs: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Dense similarity search (using vector store or token overlap similarity fallback)."""
        if not docs:
            return []

        # Simple semantic similarity proxy for dense vector matching
        query_kw = set(extract_keywords(query))
        scored_docs = []
        for doc in docs:
            doc_kw = set(extract_keywords(doc["text"]))
            inter = len(query_kw.intersection(doc_kw))
            union = len(query_kw.union(doc_kw))
            sim = (inter / float(union)) if union > 0 else 0.0
            doc_copy = dict(doc)
            doc_copy["dense_score"] = round(sim, 4)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d.get("dense_score", 0), reverse=True)
        return scored_docs[:top_k]

    def hybrid_search(
        self,
        query: str,
        target_domains: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Runs dense and sparse retrieval, fuses rankings using Reciprocal Rank Fusion (RRF)."""
        candidate_docs = self.multi_context_manager.get_all_documents(target_domains)
        if not candidate_docs:
            return []

        sparse_results = self.keyword_sparse_search(query, candidate_docs, top_k=top_k * 2)
        dense_results = self.dense_vector_search(query, candidate_docs, top_k=top_k * 2)

        fused = ReciprocalRankFusion.fuse_rankings([dense_results, sparse_results], k=60)
        return fused[:top_k]

    def query(
        self,
        question: str,
        target_domains: Optional[List[str]] = None,
        use_context_network: bool = True,
        top_k: int = 10,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Superior RAG query pipeline:
        1. Hybrid Search (Vector + Sparse) with Reciprocal Rank Fusion.
        2. Context Network graph expansion for seed document chunks.
        3. Long-context window packing and sliding context formatting.
        4. LLaMA model adapter response generation.
        """
        # Step 1: Hybrid Retrieval
        retrieved_docs = self.hybrid_search(question, target_domains=target_domains, top_k=top_k)
        retrieved_chunks = [d["text"] for d in retrieved_docs]
        seed_node_ids = [d["id"] for d in retrieved_docs]

        # Step 2: Context Network Expansion
        graph_context_str = ""
        graph_nodes: List[ContextNode] = []
        if use_context_network:
            net = self.multi_context_manager.get_combined_context_network(target_domains)
            graph_nodes = net.expand_context_graph(seed_node_ids, max_hops=2, max_nodes=10)
            graph_context_str = net.format_graph_context(graph_nodes)

        # Step 3: Long Context Packing
        packed_chunks = self.context_manager.pack_chunks(retrieved_chunks)
        formatted_context = self.context_manager.format_packed_context(
            packed_chunks=packed_chunks,
            graph_context=graph_context_str if use_context_network else None
        )

        # Step 4: LLaMA Model Generation
        default_sys_prompt = (
            "You are an advanced LLaMA assistant utilizing Multi-Context Superior RAG and Context Network analysis. "
            "Use the provided multi-context excerpts and relational network context to answer the question concisely and accurately."
        )
        sys_prompt = system_prompt or default_sys_prompt

        answer = self.adapter.generate(
            prompt=question,
            context=formatted_context,
            system_prompt=sys_prompt
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "packed_chunks": packed_chunks,
            "graph_nodes_count": len(graph_nodes),
            "target_domains": target_domains or list(self.multi_context_manager.domains.keys()),
            "model_used": self.adapter.model_name
        }
