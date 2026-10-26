import os
import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class ContextNode:
    """Node in the Context Network representing a document chunk, concept, or entity."""
    id: str
    text: str
    node_type: str = "chunk"  # 'chunk', 'entity', 'concept', 'domain'
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ContextEdge:
    """Edge in the Context Network representing relationships between nodes."""
    source_id: str
    target_id: str
    relation_type: str = "related"  # 'sequential', 'semantic_similarity', 'co_occurrence', 'entity_link'
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_keywords(text: str) -> Set[str]:
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = clean.split()
    stop_words = {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "with", "on", "for", "at", "by", "from",
        "is", "it", "this", "that", "these", "those", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "as", "if"
    }
    return {w for w in words if w not in stop_words and len(w) > 2}


class ContextNetwork:
    """
    Context Network graph engine that builds semantic relationships across chunks,
    documents, and multi-context domains, enabling multi-hop graph context retrieval.
    """

    def __init__(self):
        self.nodes: Dict[str, ContextNode] = {}
        self.edges: List[ContextEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, float, str]]] = {}  # node_id -> [(neighbor_id, weight, relation_type)]

    def add_node(
        self,
        node_id: str,
        text: str,
        node_type: str = "chunk",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> ContextNode:
        if node_id in self.nodes:
            # Update text and metadata
            self.nodes[node_id].text = text
            if metadata:
                self.nodes[node_id].metadata.update(metadata)
            if embedding:
                self.nodes[node_id].embedding = embedding
            return self.nodes[node_id]

        node = ContextNode(
            id=node_id,
            text=text,
            node_type=node_type,
            metadata=metadata or {},
            embedding=embedding
        )
        self.nodes[node_id] = node
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related",
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if source_id not in self.nodes or target_id not in self.nodes:
            return

        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata or {}
        )
        self.edges.append(edge)
        self.adjacency.setdefault(source_id, []).append((target_id, weight, relation_type))
        self.adjacency.setdefault(target_id, []).append((source_id, weight, relation_type))

    def build_network_from_chunks(
        self,
        chunks: List[str],
        doc_id: str = "doc_1",
        metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.25
    ) -> List[str]:
        """
        Builds graph nodes for document chunks, links sequential chunks, and detects co-occurring entity/keyword edges.
        Returns list of created chunk node IDs.
        """
        created_ids = []
        base_meta = metadata or {}
        keywords_per_chunk: List[Tuple[str, Set[str]]] = []

        for idx, chunk in enumerate(chunks):
            node_id = f"{doc_id}_chunk_{idx}"
            chunk_meta = {**base_meta, "doc_id": doc_id, "chunk_index": idx}
            self.add_node(node_id, text=chunk, node_type="chunk", metadata=chunk_meta)
            created_ids.append(node_id)

            kw = _extract_keywords(chunk)
            keywords_per_chunk.append((node_id, kw))

        # Add sequential edges
        for i in range(len(created_ids) - 1):
            curr_id = created_ids[i]
            next_id = created_ids[i + 1]
            self.add_edge(curr_id, next_id, relation_type="sequential", weight=1.0)

        # Add keyword / co-occurrence edges
        for i in range(len(keywords_per_chunk)):
            id_a, kw_a = keywords_per_chunk[i]
            if not kw_a:
                continue
            for j in range(i + 1, len(keywords_per_chunk)):
                id_b, kw_b = keywords_per_chunk[j]
                if not kw_b:
                    continue
                intersection = kw_a.intersection(kw_b)
                union = kw_a.union(kw_b)
                jaccard_sim = len(intersection) / float(len(union)) if union else 0.0
                if jaccard_sim >= similarity_threshold:
                    self.add_edge(
                        id_a,
                        id_b,
                        relation_type="keyword_co-occurrence",
                        weight=round(jaccard_sim, 3),
                        metadata={"shared_keywords": list(intersection)}
                    )

        return created_ids

    def expand_context_graph(
        self,
        seed_node_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 15,
        min_weight: float = 0.1
    ) -> List[ContextNode]:
        """
        Performs multi-hop graph context expansion starting from seed nodes.
        Returns list of ContextNode objects ordered by relevance / graph proximity.
        """
        visited: Set[str] = set()
        expanded_nodes: List[ContextNode] = []
        queue: List[Tuple[str, int]] = []

        for seed_id in seed_node_ids:
            if seed_id in self.nodes and seed_id not in visited:
                visited.add(seed_id)
                expanded_nodes.append(self.nodes[seed_id])
                queue.append((seed_id, 0))

        while queue and len(expanded_nodes) < max_nodes:
            curr_id, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            neighbors = self.adjacency.get(curr_id, [])
            # Sort neighbors by edge weight descending
            sorted_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

            for nbr_id, weight, rel_type in sorted_neighbors:
                if weight < min_weight:
                    continue
                if nbr_id not in visited and nbr_id in self.nodes:
                    visited.add(nbr_id)
                    expanded_nodes.append(self.nodes[nbr_id])
                    queue.append((nbr_id, hop + 1))
                    if len(expanded_nodes) >= max_nodes:
                        break

        return expanded_nodes

    def format_graph_context(self, nodes: List[ContextNode]) -> str:
        """Formats graph context nodes into readable prompt text with structural relationships."""
        if not nodes:
            return ""

        lines = ["--- Context Network Nodes & Relationships ---"]
        node_ids = {n.id for n in nodes}

        for n in nodes:
            doc = n.metadata.get("doc_id", "Unknown")
            lines.append(f"Node [{n.id}] (Doc: {doc}, Type: {n.node_type}):")
            lines.append(f"  Text: {n.text[:300]}..." if len(n.text) > 300 else f"  Text: {n.text}")

            # Highlight relevant connections among returned nodes
            connected = []
            for nbr_id, weight, rel in self.adjacency.get(n.id, []):
                if nbr_id in node_ids:
                    connected.append(f"{nbr_id} ({rel}, w={weight})")
            if connected:
                lines.append(f"  Connected to: {', '.join(connected)}")
            lines.append("")

        return "\n".join(lines)

    def save_to_json(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()

        for nd in data.get("nodes", []):
            node = ContextNode(**nd)
            self.nodes[node.id] = node
            self.adjacency[node.id] = []

        for ed in data.get("edges", []):
            edge = ContextEdge(**ed)
            self.edges.append(edge)
            self.adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge.weight, edge.relation_type))
            self.adjacency.setdefault(edge.target_id, []).append((edge.source_id, edge.weight, edge.relation_type))
