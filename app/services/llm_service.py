from typing import Any, Dict, List, Optional
from openai import OpenAI
from app.prompts import DEFAULT_RAG_SYSTEM_PROMPT


class LLMService:
    """Universal LLM Service for any OpenAI-compatible Chat Completion API."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("LLM API key is required")

        self.model_name = model_name

        # If base_url is provided (and it's not empty), use it,
        # otherwise the SDK defaults to the OpenAI API.
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url and base_url.strip():
            client_kwargs["base_url"] = base_url.strip()

        self.client = OpenAI(**client_kwargs)

    def generate_answer(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        chunks = context_chunks if context_chunks is not None else (contexts or [])

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            return {
                "answer": "Unfortunately the information was not found in the according document.",
                "sources": [],
            }

        formatted_context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", chunk)
            filename = metadata.get("source_filename") or chunk.get(
                "source_filename", "Unknown"
            )
            page = metadata.get("page_number") or chunk.get("page_number")

            page_str = f" (Page {page})" if page else ""
            formatted_context_blocks.append(
                f"[{idx}] Source: {filename}{page_str}\n{content}"
            )

        context_str = "\n\n".join(formatted_context_blocks)
        system_prompt = DEFAULT_RAG_SYSTEM_PROMPT
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer from LLM: {e}") from e

        answer_text = response.choices[0].message.content
        sources = self._extract_unique_sources(chunks)

        return {
            "answer": answer_text,
            "sources": sources,
        }

    def _extract_unique_sources(
        self, context_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique_sources = []

        for chunk in context_chunks:
            metadata = chunk.get("metadata", chunk)
            filename = metadata.get("source_filename") or chunk.get(
                "source_filename"
            )
            page_number = metadata.get("page_number") or chunk.get(
                "page_number"
            )

            source_key = (filename, page_number)
            if source_key not in seen:
                seen.add(source_key)
                unique_sources.append({
                    "source_filename": filename,
                    "page_number": page_number,
                })

        return unique_sources