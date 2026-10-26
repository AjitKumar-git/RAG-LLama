from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .adapter import estimate_tokens


@dataclass
class ContextWindowConfig:
    """Configuration for context window budget and long-context processing."""
    max_context_tokens: int = 128000
    reserved_response_tokens: int = 2048
    reserved_prompt_tokens: int = 1024
    chunk_overlap: int = 200
    target_chunk_tokens: int = 1000
    enable_compression: bool = True
    compression_ratio_threshold: float = 0.85


class LongContextManager:
    """Manages long-context packing, window slicing, token budgeting, and context compression for LLaMA models."""

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        self.config = config or ContextWindowConfig()

    def get_available_context_budget(self) -> int:
        """Returns maximum tokens allocated for context chunks after reserving overhead."""
        budget = (
            self.config.max_context_tokens
            - self.config.reserved_response_tokens
            - self.config.reserved_prompt_tokens
        )
        return max(512, budget)

    def pack_chunks(self, chunks: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """Packs as many retrieved document chunks as fit within the token budget."""
        budget = max_tokens or self.get_available_context_budget()
        packed_chunks: List[str] = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)
            if current_tokens + chunk_tokens <= budget:
                packed_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # If compression is enabled and chunk can be trimmed/summarized, compress it
                if self.config.enable_compression:
                    remaining_budget = budget - current_tokens
                    if remaining_budget >= 100:
                        trimmed = self._truncate_chunk_to_tokens(chunk, remaining_budget)
                        if trimmed:
                            packed_chunks.append(trimmed)
                break

        return packed_chunks

    def _truncate_chunk_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        max_words = max_tokens * 3  # Approximate 1 token ~ 0.75 words
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    def create_sliding_windows(self, text: str, window_tokens: int = 4000, overlap_tokens: int = 400) -> List[str]:
        """Splits long texts into sliding window context blocks."""
        words = text.split()
        window_words = window_tokens * 3
        overlap_words = overlap_tokens * 3
        step = max(1, window_words - overlap_words)

        windows = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + window_words])
            if chunk.strip():
                windows.append(chunk)
        return windows

    def format_packed_context(self, packed_chunks: List[str], graph_context: Optional[str] = None) -> str:
        """Formats packed document chunks and optional graph network context into a structured context string."""
        context_parts = []
        if graph_context:
            context_parts.append(f"=== Context Network & Relationships ===\n{graph_context}\n")

        context_parts.append("=== Document Excerpts ===")
        for idx, chunk in enumerate(packed_chunks, 1):
            context_parts.append(f"[Source {idx}]:\n{chunk}")

        return "\n\n".join(context_parts)

    def summarize_overflow_context(self, chunks: List[str], target_summary_tokens: int = 500) -> str:
        """Generates a compressed textual summary of chunks that could not fit into the context window."""
        combined = " ".join(chunks)
        words = combined.split()
        target_words = target_summary_tokens * 3
        if len(words) <= target_words:
            return combined
        return " ".join(words[:target_words]) + f"\n...[Compressed summary of {len(chunks)} overflow chunks]"
