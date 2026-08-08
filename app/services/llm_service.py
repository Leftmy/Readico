from typing import Any, Dict, List, Optional
from openai import OpenAI
from app.prompts import DEFAULT_RAG_SYSTEM_PROMPT

class LLMService:
    """Service for generating RAG answers using OpenAI Chat Completion API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        if not api_key:
            raise ValueError("API key for OpenAI is required")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate an answer grounded in the provided context chunks.

        Args:
            query (str): User question.
            context_chunks (List[Dict[str, Any]]): Retrieved document chunks with metadata.

        Returns:
            Dict[str, Any]: Containing 'answer' text and 'sources' list.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not context_chunks:
            return {
                "answer": "Unfortunately the information was not found in the according document.",
                "sources": [],
            }

        # 1. Format context chunks into a single string with citations
        formatted_context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            filename = metadata.get("source_filename", "Unknown")
            page = metadata.get("page_number")
            page_str = f" (Page {page})" if page else ""
            formatted_context_blocks.append(
                f"[{idx}] Source: {filename}{page_str}\n{content}"
            )

        context_str = "\n\n".join(formatted_context_blocks)

        # 2. System and user prompts
        system_prompt = DEFAULT_RAG_SYSTEM_PROMPT

        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # 3. Call OpenAI Chat Completion API
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

        # 4. Extracting unique sources
        sources = self._extract_unique_sources(context_chunks)

        return {
            "answer": answer_text,
            "sources": sources,
        }

    def _extract_unique_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique sources with metadata from context chunks."""
        seen = set()
        unique_sources = []

        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("source_filename")
            page_number = metadata.get("page_number")

            source_key = (filename, page_number)
            if source_key not in seen:
                seen.add(source_key)
                unique_sources.append({
                    "source_filename": filename,
                    "page_number": page_number,
                })

        return unique_sources