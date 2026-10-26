import json
import pytest
from rag_llama.mcp import RAGLlamaMCPServer
from rag_llama.rag import SuperiorRAG, MultiContextManager


def test_mcp_initialize():
    server = RAGLlamaMCPServer()
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "rag-llama-mcp-server"


def test_mcp_tools_list_and_call():
    server = RAGLlamaMCPServer()

    # Ingest document first
    server.skill.execute_action(
        "ingest_documents",
        domain_id="default",
        doc_id="mcp_doc",
        chunks=["MCP protocol connects LLM tools to RAG-LLama."]
    )

    # Tools list
    req_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp_tools = server.handle_request(req_tools)
    tools = resp_tools["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "rag_query" in tool_names
    assert "get_model_config" in tool_names

    # Tools call: get_model_config
    req_call = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "get_model_config", "arguments": {}}
    }
    resp_call = server.handle_request(req_call)
    content = resp_call["result"]["content"]
    assert len(content) > 0
    config_dict = json.loads(content[0]["text"])
    assert "model_name" in config_dict


def test_mcp_resources_and_prompts():
    server = RAGLlamaMCPServer()

    # Resources list & read
    req_res_list = {"jsonrpc": "2.0", "id": 4, "method": "resources/list"}
    resp_res_list = server.handle_request(req_res_list)
    assert len(resp_res_list["result"]["resources"]) > 0

    req_res_read = {"jsonrpc": "2.0", "id": 5, "method": "resources/read", "params": {"uri": "rag://system-status"}}
    resp_res_read = server.handle_request(req_res_read)
    assert "contents" in resp_res_read["result"]

    # Prompts list & get
    req_pr_list = {"jsonrpc": "2.0", "id": 6, "method": "prompts/list"}
    resp_pr_list = server.handle_request(req_pr_list)
    assert len(resp_pr_list["result"]["prompts"]) > 0

    req_pr_get = {"jsonrpc": "2.0", "id": 7, "method": "prompts/get", "params": {"name": "rag_system_prompt"}}
    resp_pr_get = server.handle_request(req_pr_get)
    assert "messages" in resp_pr_get["result"]
