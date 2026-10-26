from typing import Optional, List, Any
from rag_llama.models import create_llama_adapter, OllamaLlamaAdapter
from rag_llama.rag import SuperiorRAG, MultiContextManager

# Define paths
vector_store_folder = 'vector_store'


class QAChainWrapper:
    """Wrapper to maintain compatibility with legacy qa_chain.run(chain_input)."""

    def __init__(self, rag_engine: SuperiorRAG):
        self.rag_engine = rag_engine

    def run(self, chain_input: dict) -> str:
        question = chain_input.get("question", "")
        res = self.rag_engine.query(question=question)
        return res.get("answer", "")


def build_qa_chain(vectorstore=None, llm=None):
    model_name = getattr(llm, "model", "llama3.1") if llm else "llama3.1"
    adapter = create_llama_adapter(provider="ollama", model_name=model_name)
    rag_engine = SuperiorRAG(llama_adapter=adapter)
    return QAChainWrapper(rag_engine)


def query_rag(model_name: str = "llama3.1"):
    adapter = create_llama_adapter(provider="ollama", model_name=model_name)
    rag_engine = SuperiorRAG(llama_adapter=adapter)

    print(f"RAG-LLama Multi-Context System initialized (Model: {model_name}, Max Context: {adapter.get_max_context_length()} tokens)")

    while True:
        try:
            query = input("\nEnter your query (type 'quit' to exit): ")
        except EOFError:
            break
        if not query or query.lower() == 'quit':
            break

        res = rag_engine.query(question=query)
        print("Answer:", res.get("answer", ""))


if __name__ == "__main__":
    query_rag()
