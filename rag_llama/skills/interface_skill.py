from typing import Dict, List, Any, Optional
from ..rag.superior_rag import SuperiorRAG
from ..rag.multi_context import MultiContextManager


class RAGLlamaInterfaceSkill:
    """
    Interface Skill for RAG-LLama enabling agentic systems to interact with
    multi-context stores, long-context models, and the Context Network.
    """

    def __init__(self, rag_engine: Optional[SuperiorRAG] = None):
        self.rag_engine = rag_engine or SuperiorRAG()

    def get_skill_manifest(self) -> Dict[str, Any]:
        """Returns skill specification and action definitions compliant with agentic tool standards."""
        return {
            "name": "rag_llama_interface_skill",
            "description": "Multi-context and long-context RAG skill powered by generalized LLaMA adapters and Context Network.",
            "version": "2.0.0",
            "actions": {
                "rag_query": {
                    "description": "Query the RAG system across multi-context domains with Context Network enrichment.",
                    "parameters": {
                        "question": {"type": "string", "description": "The user query or prompt."},
                        "target_domains": {"type": "array", "items": {"type": "string"}, "description": "Optional domain IDs to query."},
                        "use_context_network": {"type": "boolean", "default": True, "description": "Whether to expand graph context."},
                        "top_k": {"type": "integer", "default": 10, "description": "Number of retrieved chunks."}
                    },
                    "required": ["question"]
                },
                "ingest_documents": {
                    "description": "Ingest document chunks into a specific multi-context domain and build graph network edges.",
                    "parameters": {
                        "domain_id": {"type": "string", "description": "Target domain ID."},
                        "doc_id": {"type": "string", "description": "Document ID."},
                        "chunks": {"type": "array", "items": {"type": "string"}, "description": "List of text chunks."},
                        "metadata": {"type": "object", "description": "Optional document metadata."}
                    },
                    "required": ["domain_id", "doc_id", "chunks"]
                },
                "explore_context_network": {
                    "description": "Traverse and inspect Context Network nodes and graph relationships.",
                    "parameters": {
                        "seed_node_ids": {"type": "array", "items": {"type": "string"}, "description": "Seed chunk IDs."},
                        "max_hops": {"type": "integer", "default": 2},
                        "target_domains": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["seed_node_ids"]
                },
                "manage_context_domains": {
                    "description": "List or create context domains.",
                    "parameters": {
                        "action": {"type": "string", "enum": ["list", "create"]},
                        "domain_id": {"type": "string", "description": "Required if action is create."},
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["action"]
                }
            }
        }

    def execute_action(self, action_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Routes skill invocations to appropriate skill handlers."""
        if action_name == "rag_query":
            return self._action_rag_query(**kwargs)
        elif action_name == "ingest_documents":
            return self._action_ingest_documents(**kwargs)
        elif action_name == "explore_context_network":
            return self._action_explore_context_network(**kwargs)
        elif action_name == "manage_context_domains":
            return self._action_manage_context_domains(**kwargs)
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action_name}. Available actions: {list(self.get_skill_manifest()['actions'].keys())}"
            }

    def _action_rag_query(
        self,
        question: str,
        target_domains: Optional[List[str]] = None,
        use_context_network: bool = True,
        top_k: int = 10,
        **kwargs: Any
    ) -> Dict[str, Any]:
        result = self.rag_engine.query(
            question=question,
            target_domains=target_domains,
            use_context_network=use_context_network,
            top_k=top_k
        )
        return {"status": "success", "data": result}

    def _action_ingest_documents(
        self,
        domain_id: str,
        doc_id: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        self.rag_engine.multi_context_manager.add_document_to_domain(
            domain_id=domain_id,
            doc_id=doc_id,
            text_chunks=chunks,
            metadata=metadata
        )
        return {
            "status": "success",
            "message": f"Ingested {len(chunks)} chunks into domain '{domain_id}' for doc '{doc_id}'."
        }

    def _action_explore_context_network(
        self,
        seed_node_ids: List[str],
        max_hops: int = 2,
        target_domains: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        net = self.rag_engine.multi_context_manager.get_combined_context_network(target_domains)
        expanded_nodes = net.expand_context_graph(seed_node_ids=seed_node_ids, max_hops=max_hops)
        formatted = net.format_graph_context(expanded_nodes)
        return {
            "status": "success",
            "node_count": len(expanded_nodes),
            "formatted_graph": formatted,
            "nodes": [{"id": n.id, "text": n.text[:100], "type": n.node_type} for n in expanded_nodes]
        }

    def _action_manage_context_domains(
        self,
        action: str,
        domain_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        mcm = self.rag_engine.multi_context_manager
        if action == "list":
            return {"status": "success", "domains": mcm.list_domains()}
        elif action == "create":
            if not domain_id:
                return {"status": "error", "message": "domain_id is required to create a domain"}
            domain = mcm.create_domain(domain_id=domain_id, name=name or domain_id, description=description or "")
            return {"status": "success", "domain_id": domain.domain_id, "name": domain.name}
        else:
            return {"status": "error", "message": f"Invalid action: {action}"}
