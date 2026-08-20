from app.core.config import settings
import cohere
from typing import List, Tuple


co = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)

# DOCUMENT RERANKER (unchanged logic, kept separate from the tool itself)
class DocumentReranker:
    """Wraps Cohere reranking logic."""

    def __init__(self, cohere_client, model: str = "rerank-v4.0-fast"):
        self.co = cohere_client
        self.model = model

    async def rerank(self, docs: list, query: str, top_n: int) -> List[Tuple[str, float]]:
        doc_texts = [doc.page_content.strip() for doc in docs]

        rerank_results = await self.co.rerank(
            query=query,
            documents=doc_texts,
            top_n=min(top_n, len(doc_texts)),
            model=self.model,
        )

        seen = set()
        output = []
        results_list = getattr(rerank_results, "results", rerank_results)
        for res in results_list:
            text = doc_texts[res.index]
            if text not in seen:
                seen.add(text)
                output.append((text, res.relevance_score))

        return output


reranker = DocumentReranker(cohere_client=co)



