import pytest
from rag_llama.rag import SuperiorRAG, MultiContextManager, ReciprocalRankFusion
from rag_llama.models import create_llama_adapter


def test_reciprocal_rank_fusion():
    list1 = [{"id": "doc1", "text": "A"}, {"id": "doc2", "text": "B"}]
    list2 = [{"id": "doc2", "text": "B"}, {"id": "doc3", "text": "C"}]

    fused = ReciprocalRankFusion.fuse_rankings([list1, list2], k=60)
    assert len(fused) == 3
    # doc2 appears in both lists, so its RRF score should be highest
    assert fused[0]["id"] == "doc2"


def test_multi_context_manager():
    mcm = MultiContextManager()
    mcm.create_domain("tech", "Technical Docs")
    mcm.create_domain("finance", "Financial Reports")

    mcm.add_document_to_domain("tech", "doc_tech", ["LLaMA architecture and long context windows."])
    mcm.add_document_to_domain("finance", "doc_fin", ["Quarterly revenue growth and balance sheet."])

    all_docs = mcm.get_all_documents()
    assert len(all_docs) == 2

    tech_docs = mcm.get_all_documents(target_domains=["tech"])
    assert len(tech_docs) == 1
    assert tech_docs[0]["doc_id"] == "doc_tech"


def test_superior_rag_query():
    mcm = MultiContextManager()
    mcm.add_document_to_domain(
        "default",
        "doc1",
        ["LLaMA 3.1 supports 128,000 token context windows.", "RAG-LLama provides multi-context and graph retrieval."]
    )

    adapter = create_llama_adapter("ollama", "llama3.1")
    rag = SuperiorRAG(llama_adapter=adapter, multi_context_manager=mcm)

    res = rag.query("What context size does LLaMA 3.1 support?")
    assert res["question"] == "What context size does LLaMA 3.1 support?"
    assert len(res["retrieved_chunks"]) > 0
    assert res["model_used"] == "llama3.1"
