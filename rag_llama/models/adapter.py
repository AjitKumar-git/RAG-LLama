import os
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


# Standard LLaMA Model Context Window Specs
LLAMA_CONTEXT_WINDOWS = {
    "llama3.3": 128000,
    "llama3.2": 128000,
    "llama3.1": 128000,
    "llama3": 8192,
    "llama2": 4096,
    "codellama": 16384,
    "vicuna": 4096,
    "alpaca": 4096,
    "mistral": 32768,
    "mixtral": 32768,
}


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text (avg ~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class BaseLlamaAdapter(ABC):
    """Abstract base adapter for all LLaMA-based models."""

    def __init__(self, model_name: str = "llama3.1", max_context_length: Optional[int] = None):
        self.model_name = model_name.lower()
        if max_context_length:
            self.max_context_length = max_context_length
        else:
            self.max_context_length = self._infer_max_context(self.model_name)

    def _infer_max_context(self, model_name: str) -> int:
        for key, val in LLAMA_CONTEXT_WINDOWS.items():
            if key in model_name:
                return val
        return 128000  # Default long context for modern LLaMA models

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def get_max_context_length(self) -> int:
        return self.max_context_length

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate text from prompt and optional context."""
        pass


class OllamaLlamaAdapter(BaseLlamaAdapter):
    """Adapter for Ollama running LLaMA models."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        max_context_length: Optional[int] = None
    ):
        super().__init__(model_name=model_name, max_context_length=max_context_length)
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion/Task:\n{prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.max_context_length,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "")
        except Exception as e:
            # Standalone fallback generation when server is unreachable or offline
            return self._fallback_response(prompt, context, str(e))

    def _fallback_response(self, prompt: str, context: Optional[str], error_msg: str) -> str:
        ctx_summary = f" (Context size: {len(context)} chars)" if context else ""
        return (
            f"[RAG-LLama ({self.model_name})] Offline/Mock response for: '{prompt[:50]}...'{ctx_summary}\n"
            f"(Note: Could not reach Ollama at {self.base_url}: {error_msg})"
        )


class OpenAILlamaAdapter(BaseLlamaAdapter):
    """Adapter for OpenAI-compatible LLaMA server endpoints (vLLM, LM Studio, LocalAI)."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_context_length: Optional[int] = None
    ):
        super().__init__(model_name=model_name, max_context_length=max_context_length)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}" if context else prompt
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[RAG-LLama OpenAI-Compatible ({self.model_name})] Fallback response: {e}"


class HuggingFaceLlamaAdapter(BaseLlamaAdapter):
    """Adapter for HuggingFace LLaMA endpoints or local HF pipelines."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_token: Optional[str] = None,
        max_context_length: Optional[int] = None
    ):
        super().__init__(model_name=model_name, max_context_length=max_context_length)
        self.api_token = api_token or os.getenv("HF_TOKEN", "")

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        if not self.api_token:
            return f"[RAG-LLama HuggingFace ({self.model_name})] Prompt formatted ({len(full_prompt)} chars). Token required for live HF API call."

        payload = {
            "inputs": full_prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
        }
        try:
            url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                if isinstance(res_data, list) and len(res_data) > 0:
                    return res_data[0].get("generated_text", "")
                return str(res_data)
        except Exception as e:
            return f"[RAG-LLama HuggingFace ({self.model_name})] Inference error: {e}"


def create_llama_adapter(
    provider: str = "ollama",
    model_name: str = "llama3.1",
    max_context_length: Optional[int] = None,
    **kwargs: Any
) -> BaseLlamaAdapter:
    """Factory function to instantiate LLaMA model adapters."""
    provider = provider.lower()
    if provider == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaLlamaAdapter(model_name=model_name, base_url=base_url, max_context_length=max_context_length)
    elif provider in ("openai", "openai_compatible", "vllm", "lmstudio"):
        api_base = kwargs.get("api_base", "http://localhost:8000/v1")
        api_key = kwargs.get("api_key", "EMPTY")
        return OpenAILlamaAdapter(model_name=model_name, api_base=api_base, api_key=api_key, max_context_length=max_context_length)
    elif provider in ("hf", "huggingface"):
        api_token = kwargs.get("api_token", None)
        return HuggingFaceLlamaAdapter(model_name=model_name, api_token=api_token, max_context_length=max_context_length)
    else:
        raise ValueError(f"Unsupported LLaMA provider: {provider}")
