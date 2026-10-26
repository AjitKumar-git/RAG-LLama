import re
from typing import List, Optional, Any
from rag_llama.models import create_llama_adapter
from rag_llama.rag import SuperiorRAG, MultiContextManager
from rag_llama.network import ContextNetwork

# Define paths
vector_store_folder = 'vector_store'


def extract_keywords(query: str) -> List[str]:
    """Extracts keywords from input query removing stop words and non-alphanumeric chars."""
    query = query.lower()
    query = re.sub(r'[^a-z0-9\s]', '', query)
    stop_words = set(["a", "an", "the", "and", "or", "but", "of", "to", "in", "with", "on", "for", "at", "by"])
    return [word for word in query.split() if word not in stop_words]


class QAChainWrapperV2:
    """Wrapper to maintain compatibility with legacy qa_chain.run(chain_input)."""

    def __init__(self, rag_engine: SuperiorRAG):
        self.rag_engine = rag_engine

    def run(self, chain_input: dict) -> str:
        question = chain_input.get("question", "")
        res = self.rag_engine.query(question=question, use_context_network=True)
        return res.get("answer", "")


def build_qa_chain(vectorstore=None, llm=None):
    model_name = getattr(llm, "model", "llama3.1") if llm else "llama3.1"
    adapter = create_llama_adapter(provider="ollama", model_name=model_name)
    rag_engine = SuperiorRAG(llama_adapter=adapter)
    return QAChainWrapperV2(rag_engine)


def query_rag(model_name: str = "llama3.1"):
    adapter = create_llama_adapter(provider="ollama", model_name=model_name)
    rag_engine = SuperiorRAG(llama_adapter=adapter)

    print(f"RAG-LLama Superior RAG & Context Network initialized (Model: {model_name}, Max Context: {adapter.get_max_context_length()} tokens)")

    while True:
        try:
            query = input("\nEnter your query (type 'quit' to exit): ")
        except EOFError:
            break
        if not query or query.lower() == 'quit':
            break

        keywords = extract_keywords(query)
        res = rag_engine.query(question=query, use_context_network=True)
        print("Answer:", res.get("answer", ""))


if __name__ == "__main__":
    query_rag()
