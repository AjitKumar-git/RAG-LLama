import pytest
from rag_llama.models import create_llama_adapter, OllamaLlamaAdapter, OpenAILlamaAdapter, HuggingFaceLlamaAdapter, LongContextManager, ContextWindowConfig


def test_llama_adapter_factory():
    ollama_adapter = create_llama_adapter("ollama", "llama3.1")
    assert isinstance(ollama_adapter, OllamaLlamaAdapter)
    assert ollama_adapter.get_max_context_length() == 128000

    openai_adapter = create_llama_adapter("openai", "llama3")
    assert isinstance(openai_adapter, OpenAILlamaAdapter)
    assert openai_adapter.get_max_context_length() == 8192

    hf_adapter = create_llama_adapter("hf", "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(hf_adapter, HuggingFaceLlamaAdapter)


def test_long_context_manager_packing():
    config = ContextWindowConfig(max_context_tokens=1000, reserved_response_tokens=100, reserved_prompt_tokens=100)
    cm = LongContextManager(config=config)
    assert cm.get_available_context_budget() == 800

    chunks = ["Short chunk " + str(i) for i in range(20)]
    packed = cm.pack_chunks(chunks)
    assert len(packed) > 0
    assert len(packed) <= len(chunks)


def test_sliding_window_creation():
    cm = LongContextManager()
    long_text = " ".join([f"word{i}" for i in range(3000)])
    windows = cm.create_sliding_windows(long_text, window_tokens=200, overlap_tokens=20)
    assert len(windows) > 1
