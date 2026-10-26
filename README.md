# RAG-LLama

**RAG-LLama** is an enterprise-grade Retrieval-Augmented Generation (RAG) framework generalized for **all LLaMA-based systems**, featuring **Multi-Context management**, **Long-Context window optimization**, **Context Network graph expansion**, **Interface Skills**, and a standard **Model Context Protocol (MCP) Server**.

---

## 🌟 Key Features

1. **Generalized LLaMA Model Architecture (`rag_llama.models`)**
   - Supports all LLaMA model variants: **LLaMA 3.3**, **LLaMA 3.2**, **LLaMA 3.1**, **LLaMA 3**, **CodeLlama**, **LLaMA 2**, **Vicuna**, **Mistral**, etc.
   - Provider adapters for **Ollama**, **OpenAI-compatible APIs** (vLLM, LM Studio, LocalAI), and **Hugging Face Inference**.
   - **Long-Context Window Optimization**: Dynamic token budgeting, context packing up to **128,000+ tokens**, sliding window chunking, and context summarization.

2. **Multi-Context Domain Management (`rag_llama.rag.multi_context`)**
   - Isolate, combine, and route queries across multiple document domains, vector stores, and knowledge bases.
   - Cross-domain context aggregation and filtering.

3. **Superior RAG Engine & Context Network (`rag_llama.rag.superior_rag` & `rag_llama.network`)**
   - **Hybrid Retrieval**: Dense similarity search combined with sparse BM25/keyword retrieval.
   - **Reciprocal Rank Fusion (RRF)**: Merges rank results from multiple search strategies for top-k selection.
   - **Context Network Graph**: Knowledge graph structure linking document chunks, co-occurring concepts, and sequential sections with multi-hop graph expansion (`expand_context_graph`) to enrich retrieved LLM context.

4. **Interface Skill (`rag_llama.skills`)**
   - Standardized AI agent skill definitions and action routing for integration with agent frameworks (e.g. Agency, OpenAI tool schemas).
   - Exposed actions: `rag_query`, `ingest_documents`, `explore_context_network`, `manage_context_domains`.

5. **Model Context Protocol (MCP) Server (`rag_llama.mcp`)**
   - Native **MCP Server** implementation supporting JSON-RPC 2.0 over stdio/HTTP.
   - Exposes RAG tools (`rag_query`, `ingest_file`, `explore_context_network`, `manage_contexts`, `get_model_config`), context resources (`context://domains`, `context://networks`, `rag://system-status`), and system prompts (`rag_system_prompt`, `long_context_qa`).

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt # or install pytest, ebooklib, beautifulsoup4, pypdf2
```

### CLI Command Options (`main.py`)

#### 1. Execute a Multi-Context RAG Query

```bash
python main.py --query "What are the context length capabilities of LLaMA 3.1?" --model llama3.1 --provider ollama
```

#### 2. Specify Model Variant and Context Window

```bash
python main.py --query "Analyze codebase architecture" --model codellama --context-window 32768
```

#### 3. Ingest Documents into Multi-Context Domains

```bash
python main.py --ingest ./input_files --domain technical_docs
```

#### 4. Launch the Model Context Protocol (MCP) Server

```bash
python main.py --mcp-server
```

#### 5. Execute an Interface Skill Action via JSON

```bash
python main.py --skill-action '{"action": "rag_query", "question": "Explain graph RAG", "target_domains": ["technical_docs"]}'
```

#### 6. Inspect Context Network Topology

```bash
python main.py --inspect-network
```

---

## 💻 Python API Usage

### Querying with Superior RAG & LLaMA Adapters

```python
from rag_llama.models import create_llama_adapter
from rag_llama.rag import SuperiorRAG, MultiContextManager

# 1. Initialize generalized LLaMA adapter (e.g. LLaMA 3.1 128k context)
adapter = create_llama_adapter(provider="ollama", model_name="llama3.1")

# 2. Setup Multi-Context Manager & ingest documents
mcm = MultiContextManager()
mcm.add_document_to_domain(
    domain_id="ai_research",
    doc_id="paper1",
    chunks=["LLaMA 3.1 supports long context up to 128k tokens.", "Context Network expands context graphs."]
)

# 3. Create Superior RAG Engine
rag = SuperiorRAG(llama_adapter=adapter, multi_context_manager=mcm)

# 4. Execute query with hybrid retrieval & Context Network graph expansion
response = rag.query("What context length does LLaMA 3.1 support?")
print("Answer:", response["answer"])
```

### Context Network Multi-Hop Expansion

```python
from rag_llama.network import ContextNetwork

net = ContextNetwork()
net.build_network_from_chunks(["Chunk 1 text...", "Chunk 2 related text..."], doc_id="doc1")

# Expand graph context starting from seed chunk
expanded_nodes = net.expand_context_graph(seed_node_ids=["doc1_chunk_0"], max_hops=2)
formatted_graph = net.format_graph_context(expanded_nodes)
print(formatted_graph)
```

### Interface Skill Integration

```python
from rag_llama.skills import RAGLlamaInterfaceSkill

skill = RAGLlamaInterfaceSkill()
manifest = skill.get_skill_manifest()

# Execute skill action
result = skill.execute_action(
    "rag_query",
    question="Explain long context RAG",
    use_context_network=True
)
print(result)
```

---

## 🧪 Testing

Run unit and integration tests using `pytest`:

```bash
pytest -v
```

---

## 📜 License

MIT License.
