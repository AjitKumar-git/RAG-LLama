import sys
import json
import logging
from typing import Dict, List, Any, Optional
from ..rag.superior_rag import SuperiorRAG
from ..skills.interface_skill import RAGLlamaInterfaceSkill

logger = logging.getLogger("RAGLlamaMCPServer")


class RAGLlamaMCPServer:
    """
    Model Context Protocol (MCP) server for RAG-LLama over JSON-RPC 2.0.
    Exposes RAG tools, Context Network resources, and system prompts to MCP clients.
    """

    def __init__(self, rag_engine: Optional[SuperiorRAG] = None):
        self.rag_engine = rag_engine or SuperiorRAG()
        self.skill = RAGLlamaInterfaceSkill(self.rag_engine)
        self.protocol_version = "2024-11-05"

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC request and returns a JSON-RPC response."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if not method:
            return self._make_error_response(req_id, -32600, "Invalid Request: missing method")

        try:
            if method == "initialize":
                return self._handle_initialize(req_id, params)
            elif method == "notifications/initialized":
                return None  # Notification, no response needed
            elif method == "ping":
                return self._make_result_response(req_id, {})
            elif method == "tools/list":
                return self._handle_tools_list(req_id)
            elif method == "tools/call":
                return self._handle_tools_call(req_id, params)
            elif method == "resources/list":
                return self._handle_resources_list(req_id)
            elif method == "resources/read":
                return self._handle_resources_read(req_id, params)
            elif method == "prompts/list":
                return self._handle_prompts_list(req_id)
            elif method == "prompts/get":
                return self._handle_prompts_get(req_id, params)
            else:
                return self._make_error_response(req_id, -32601, f"Method not found: {method}")
        except Exception as e:
            return self._make_error_response(req_id, -32603, f"Internal error: {str(e)}")

    def _make_result_response(self, req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _make_error_response(self, req_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message}
        }

    def _handle_initialize(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_result_response(req_id, {
            "protocolVersion": self.protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False}
            },
            "serverInfo": {
                "name": "rag-llama-mcp-server",
                "version": "2.0.0"
            }
        })

    def _handle_tools_list(self, req_id: Any) -> Dict[str, Any]:
        manifest = self.skill.get_skill_manifest()
        tools = []

        for name, spec in manifest["actions"].items():
            tools.append({
                "name": name,
                "description": spec["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": spec.get("parameters", {}),
                    "required": spec.get("required", [])
                }
            })

        # Add explicit MCP tool for get_model_config
        tools.append({
            "name": "get_model_config",
            "description": "Get current LLaMA model adapter configuration and context window budget.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        })

        return self._make_result_response(req_id, {"tools": tools})

    def _handle_tools_call(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "get_model_config":
            adapter = self.rag_engine.adapter
            config_data = {
                "model_name": adapter.model_name,
                "max_context_length": adapter.get_max_context_length(),
                "available_context_budget": self.rag_engine.context_manager.get_available_context_budget(),
                "provider": adapter.__class__.__name__
            }
            return self._make_result_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(config_data, indent=2)}]
            })

        skill_res = self.skill.execute_action(tool_name, **arguments)
        text_out = json.dumps(skill_res, indent=2)
        return self._make_result_response(req_id, {
            "content": [{"type": "text", "text": text_out}]
        })

    def _handle_resources_list(self, req_id: Any) -> Dict[str, Any]:
        resources = [
            {
                "uri": "context://domains",
                "name": "Multi-Context Domains List",
                "description": "All registered multi-context domains and document counts",
                "mimeType": "application/json"
            },
            {
                "uri": "context://networks",
                "name": "Context Network Overview",
                "description": "Graph topology node and edge count overview",
                "mimeType": "application/json"
            },
            {
                "uri": "rag://system-status",
                "name": "RAG-LLama System Status",
                "description": "RAG engine operational status and configuration",
                "mimeType": "application/json"
            }
        ]
        return self._make_result_response(req_id, {"resources": resources})

    def _handle_resources_read(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if uri == "context://domains":
            domains_info = self.rag_engine.multi_context_manager.list_domains()
            return self._make_result_response(req_id, {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(domains_info, indent=2)}]
            })
        elif uri == "context://networks":
            combined_net = self.rag_engine.multi_context_manager.get_combined_context_network()
            net_info = {
                "node_count": len(combined_net.nodes),
                "edge_count": len(combined_net.edges),
                "nodes_sample": [n.id for n in list(combined_net.nodes.values())[:10]]
            }
            return self._make_result_response(req_id, {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(net_info, indent=2)}]
            })
        elif uri == "rag://system-status":
            status = {
                "version": "2.0.0",
                "model_name": self.rag_engine.adapter.model_name,
                "max_context_tokens": self.rag_engine.adapter.get_max_context_length(),
                "active_domains": len(self.rag_engine.multi_context_manager.domains)
            }
            return self._make_result_response(req_id, {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(status, indent=2)}]
            })
        else:
            return self._make_error_response(req_id, -32602, f"Resource URI not found: {uri}")

    def _handle_prompts_list(self, req_id: Any) -> Dict[str, Any]:
        prompts = [
            {
                "name": "rag_system_prompt",
                "description": "Standard system prompt for Multi-Context Superior RAG with LLaMA models."
            },
            {
                "name": "long_context_qa",
                "description": "Prompt optimized for long-context analysis with structured Context Network graph output."
            }
        ]
        return self._make_result_response(req_id, {"prompts": prompts})

    def _handle_prompts_get(self, req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt_name = params.get("name")
        if prompt_name == "rag_system_prompt":
            return self._make_result_response(req_id, {
                "description": "Standard system prompt",
                "messages": [
                    {
                        "role": "system",
                        "content": {
                            "type": "text",
                            "text": "You are an AI assistant using RAG-LLama with Multi-Context retrieval and Context Network relationships."
                        }
                    }
                ]
            })
        elif prompt_name == "long_context_qa":
            return self._make_result_response(req_id, {
                "description": "Long context prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Analyze the following multi-context document graph and answer the question in detail."
                        }
                    }
                ]
            })
        else:
            return self._make_error_response(req_id, -32602, f"Prompt not found: {prompt_name}")

    def run_std_io(self):
        """Runs the MCP server loop reading JSON-RPC requests from stdio."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = self._make_error_response(None, -32700, f"Parse error: {str(e)}")
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = RAGLlamaMCPServer()
    server.run_std_io()
