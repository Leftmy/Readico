import logging
from typing import Any, Dict, List
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> None:
        # Load the Cross-Encoder model once on service startup
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Re-scores retrieved vector candidates using Cross-Encoder attention 
        and filters out low-relevance noise.
        """
        if not results:
            return []

        # Map vector search payload to FlashRank passage format
        passages = [
            {
                "id": idx,
                "text": item.get("content", ""),
                "meta": item,
            }
            for idx, item in enumerate(results)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)

        final_results = []
        for ranked in ranked_results:
            new_score = float(ranked["score"])

            # Filter out chunks that do not meet the strict relevance threshold
            if new_score >= score_threshold:
                item_data = ranked["meta"]
                item_data["score"] = round(new_score, 4)
                final_results.append(item_data)

        # Sort descending by Cross-Encoder score and slice top_n
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_n]