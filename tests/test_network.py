import os
import tempfile
import pytest
from rag_llama.network import ContextNetwork, ContextNode, ContextEdge


def test_context_network_nodes_and_edges():
    net = ContextNetwork()
    net.add_node("n1", "Text chunk 1", node_type="chunk")
    net.add_node("n2", "Text chunk 2", node_type="chunk")
    net.add_edge("n1", "n2", relation_type="sequential", weight=1.0)

    assert len(net.nodes) == 2
    assert len(net.edges) == 1
    assert "n2" in [nbr[0] for nbr in net.adjacency["n1"]]


def test_build_network_from_chunks():
    net = ContextNetwork()
    chunks = [
        "LLaMA 3.1 is an open-weights model developed by Meta.",
        "Meta released LLaMA 3.1 with 128k long context support.",
        "Context Network graph improves RAG retrieval accuracy."
    ]
    created_ids = net.build_network_from_chunks(chunks, doc_id="test_doc", similarity_threshold=0.1)

    assert len(created_ids) == 3
    assert len(net.nodes) == 3
    assert len(net.edges) >= 2  # At least sequential edges


def test_expand_context_graph():
    net = ContextNetwork()
    net.add_node("A", "Seed node A")
    net.add_node("B", "Neighbor B")
    net.add_node("C", "Multi-hop C")
    net.add_edge("A", "B", weight=0.8)
    net.add_edge("B", "C", weight=0.9)

    expanded = net.expand_context_graph(["A"], max_hops=2, max_nodes=5)
    expanded_ids = [n.id for n in expanded]
    assert "A" in expanded_ids
    assert "B" in expanded_ids
    assert "C" in expanded_ids


def test_save_and_load_json():
    net = ContextNetwork()
    net.add_node("n1", "Content")
    net.add_node("n2", "More content")
    net.add_edge("n1", "n2", weight=0.5)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        net.save_to_json(tmp_path)
        new_net = ContextNetwork()
        new_net.load_from_json(tmp_path)

        assert len(new_net.nodes) == 2
        assert len(new_net.edges) == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
