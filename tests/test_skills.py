import pytest
from rag_llama.skills import RAGLlamaInterfaceSkill
from rag_llama.rag import SuperiorRAG, MultiContextManager


def test_interface_skill_manifest():
    skill = RAGLlamaInterfaceSkill()
    manifest = skill.get_skill_manifest()
    assert manifest["name"] == "rag_llama_interface_skill"
    assert "rag_query" in manifest["actions"]
    assert "ingest_documents" in manifest["actions"]
    assert "explore_context_network" in manifest["actions"]
    assert "manage_context_domains" in manifest["actions"]


def test_interface_skill_actions():
    skill = RAGLlamaInterfaceSkill()

    # Ingest document action
    ingest_res = skill.execute_action(
        "ingest_documents",
        domain_id="test_domain",
        doc_id="doc1",
        chunks=["LLaMA long context models enable deep analysis."]
    )
    assert ingest_res["status"] == "success"

    # Query action
    query_res = skill.execute_action(
        "rag_query",
        question="What enables deep analysis?",
        target_domains=["test_domain"]
    )
    assert query_res["status"] == "success"
    assert "answer" in query_res["data"]

    # Explore network action
    explore_res = skill.execute_action(
        "explore_context_network",
        seed_node_ids=["doc1_chunk_0"],
        target_domains=["test_domain"]
    )
    assert explore_res["status"] == "success"
    assert explore_res["node_count"] > 0
