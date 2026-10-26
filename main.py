import os
import sys
import json
import argparse
from typing import List, Optional

from rag_llama.models import create_llama_adapter, LongContextManager
from rag_llama.rag import SuperiorRAG, MultiContextManager
from rag_llama.network import ContextNetwork
from rag_llama.skills import RAGLlamaInterfaceSkill
from rag_llama.mcp import RAGLlamaMCPServer
from convert_files import convert_files_to_text


# Backward compatibility helper classes for legacy main.py usage
class DocumentIngestor:
    """Document ingestor alias for backward compatibility."""
    def __init__(self, multi_context_manager: Optional[MultiContextManager] = None):
        self.mcm = multi_context_manager or MultiContextManager()

    def process_file(
        self,
        file_path: str,
        vector_store_path: str = "vector_store",
        record_file: str = "record.json",
        domain_id: str = "default"
    ):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        doc_id = os.path.basename(file_path)
        self.mcm.add_document_to_domain(
            domain_id=domain_id,
            doc_id=doc_id,
            text_chunks=chunks,
            metadata={"source_file": file_path, "vector_store_path": vector_store_path}
        )
        print(f"Successfully ingested '{file_path}' ({len(chunks)} chunks) into domain '{domain_id}'.")


class MultiRAGManager:
    """Multi-RAG Manager alias for backward compatibility."""
    def __init__(
        self,
        vector_store_paths: Optional[List[str]] = None,
        model_name: str = "llama3.1",
        provider: str = "ollama"
    ):
        self.mcm = MultiContextManager()
        if vector_store_paths:
            for idx, vp in enumerate(vector_store_paths):
                domain_id = f"domain_{idx+1}"
                self.mcm.create_domain(domain_id=domain_id, name=f"VectorStore {vp}", vector_store_path=vp)

        adapter = create_llama_adapter(provider=provider, model_name=model_name)
        self.rag = SuperiorRAG(llama_adapter=adapter, multi_context_manager=self.mcm)

    def query(self, question: str, target_domains: Optional[List[str]] = None) -> str:
        res = self.rag.query(question=question, target_domains=target_domains)
        return res.get("answer", "")


def main():
    parser = argparse.ArgumentParser(
        description="RAG-LLama: Multi-Context and Long-Context Generalized RAG Framework for LLaMA Systems"
    )
    parser.add_argument("--query", "-q", type=str, help="Question/Query to process with Superior RAG")
    parser.add_argument("--model", "-m", type=str, default="llama3.1", help="LLaMA model variant (llama3.3, llama3.2, llama3.1, llama3, codellama, llama2, etc.)")
    parser.add_argument("--provider", "-p", type=str, default="ollama", choices=["ollama", "openai", "hf"], help="LLaMA model provider adapter")
    parser.add_argument("--context-window", type=int, default=None, help="Context window token limit override (e.g., 128000)")
    parser.add_argument("--domain", "-d", type=str, action="append", help="Target domain ID(s) to search")
    parser.add_argument("--ingest", "-i", type=str, help="Ingest a document file or directory into multi-context domain")
    parser.add_argument("--mcp-server", action="store_true", help="Start Model Context Protocol (MCP) server over stdio")
    parser.add_argument("--skill-action", type=str, help="Execute an interface skill action JSON: '{\"action\": \"rag_query\", \"question\": \"...\"}'")
    parser.add_argument("--inspect-network", action="store_true", help="Inspect topology and nodes in the Context Network")

    args = parser.parse_args()

    # Launch MCP Server mode
    if args.mcp_server:
        adapter = create_llama_adapter(provider=args.provider, model_name=args.model, max_context_length=args.context_window)
        rag_engine = SuperiorRAG(llama_adapter=adapter)
        mcp_server = RAGLlamaMCPServer(rag_engine=rag_engine)
        print("Starting RAG-LLama MCP Server...", file=sys.stderr)
        mcp_server.run_std_io()
        return

    # Initialize RAG System
    adapter = create_llama_adapter(provider=args.provider, model_name=args.model, max_context_length=args.context_window)
    mcm = MultiContextManager()
    rag_engine = SuperiorRAG(llama_adapter=adapter, multi_context_manager=mcm)
    skill = RAGLlamaInterfaceSkill(rag_engine=rag_engine)

    # Ingest file if requested
    if args.ingest:
        ingestor = DocumentIngestor(multi_context_manager=mcm)
        target_domain = args.domain[0] if args.domain else "default"
        if os.path.isfile(args.ingest):
            ingestor.process_file(args.ingest, domain_id=target_domain)
        elif os.path.isdir(args.ingest):
            for fname in os.listdir(args.ingest):
                fpath = os.path.join(args.ingest, fname)
                if os.path.isfile(fpath):
                    ingestor.process_file(fpath, domain_id=target_domain)

    # Execute Interface Skill JSON if requested
    if args.skill_action:
        try:
            skill_input = json.loads(args.skill_action)
            action = skill_input.pop("action")
            result = skill.execute_action(action, **skill_input)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error executing skill action: {e}")
        return

    # Inspect Network if requested
    if args.inspect_network:
        net = mcm.get_combined_context_network()
        print(f"Context Network Summary:")
        print(f"  Total Nodes: {len(net.nodes)}")
        print(f"  Total Edges: {len(net.edges)}")
        for n_id, node in list(net.nodes.items())[:5]:
            print(f"  - Node [{n_id}]: {node.text[:80]}...")
        return

    # Run Query if requested
    if args.query:
        print(f"\n--- Processing Query using LLaMA Model: {args.model} ({adapter.get_max_context_length()} tokens window) ---")
        result = rag_engine.query(question=args.query, target_domains=args.domain)
        print("\nAnswer:\n", result["answer"])
        print(f"\n[Retrieved {len(result['retrieved_chunks'])} chunks, Context Graph Nodes: {result['graph_nodes_count']}]")
        return

    # Default fallback prompt / help if no flags passed
    if not any([args.query, args.ingest, args.skill_action, args.inspect_network, args.mcp_server]):
        parser.print_help()


if __name__ == "__main__":
    main()
