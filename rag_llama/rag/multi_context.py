import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from ..network.context_network import ContextNetwork


@dataclass
class MultiContextDomain:
    """Represents an isolated or aggregated context domain / vector store."""
    domain_id: str
    name: str
    description: str = ""
    vector_store_path: str = ""
    record_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    documents: List[Dict[str, Any]] = field(default_factory=list)  # In-memory doc cache: [{'id': ..., 'text': ..., 'metadata': ...}]


class MultiContextManager:
    """Manages multi-context domains, multi-vector-store routing, and cross-domain document retrieval."""

    def __init__(self, default_domain: str = "default"):
        self.domains: Dict[str, MultiContextDomain] = {}
        self.context_networks: Dict[str, ContextNetwork] = {}
        self.default_domain_id = default_domain

        # Create default domain
        self.create_domain(
            domain_id=default_domain,
            name="Default Domain",
            description="Default knowledge context store",
            vector_store_path="vector_store"
        )

    def create_domain(
        self,
        domain_id: str,
        name: str,
        description: str = "",
        vector_store_path: str = "",
        record_file: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> MultiContextDomain:
        domain = MultiContextDomain(
            domain_id=domain_id,
            name=name,
            description=description,
            vector_store_path=vector_store_path or f"vector_store_{domain_id}",
            record_file=record_file or f"record_{domain_id}.json",
            metadata=metadata or {}
        )
        self.domains[domain_id] = domain
        if domain_id not in self.context_networks:
            self.context_networks[domain_id] = ContextNetwork()
        return domain

    def get_domain(self, domain_id: str) -> Optional[MultiContextDomain]:
        return self.domains.get(domain_id)

    def list_domains(self) -> List[Dict[str, Any]]:
        return [
            {
                "domain_id": d.domain_id,
                "name": d.name,
                "description": d.description,
                "vector_store_path": d.vector_store_path,
                "doc_count": len(d.documents)
            }
            for d in self.domains.values()
        ]

    def add_document_to_domain(
        self,
        domain_id: str,
        doc_id: str,
        text_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        domain = self.domains.get(domain_id)
        if not domain:
            domain = self.create_domain(domain_id=domain_id, name=domain_id)

        doc_meta = metadata or {}
        for idx, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id}_c{idx}"
            domain.documents.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk,
                "metadata": {**doc_meta, "chunk_index": idx, "domain_id": domain_id}
            })

        # Build Context Network edges inside the domain network
        net = self.context_networks[domain_id]
        net.build_network_from_chunks(text_chunks, doc_id=doc_id, metadata={**doc_meta, "domain_id": domain_id})

    def get_all_documents(self, target_domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieves document chunks from specified domains or all domains if target_domains is None."""
        selected_domains = target_domains if target_domains else list(self.domains.keys())
        all_docs = []
        for d_id in selected_domains:
            domain = self.domains.get(d_id)
            if domain:
                all_docs.extend(domain.documents)
        return all_docs

    def get_combined_context_network(self, target_domains: Optional[List[str]] = None) -> ContextNetwork:
        """Merges Context Networks across specified domains into a unified ContextNetwork."""
        selected_domains = target_domains if target_domains else list(self.domains.keys())
        combined_net = ContextNetwork()
        for d_id in selected_domains:
            net = self.context_networks.get(d_id)
            if net:
                for n_id, node in net.nodes.items():
                    combined_net.add_node(node.id, text=node.text, node_type=node.node_type, metadata=node.metadata, embedding=node.embedding)
                for edge in net.edges:
                    combined_net.add_edge(edge.source_id, edge.target_id, relation_type=edge.relation_type, weight=edge.weight, metadata=edge.metadata)
        return combined_net
