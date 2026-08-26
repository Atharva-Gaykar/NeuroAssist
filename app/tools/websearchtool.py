from typing import List, Dict, Any
from serpapi import GoogleSearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
# app
from app.core.config import settings
from typing import List, Dict, Any,Tuple, Optional
#app
from app.tools.tools_input_schema import SerpSearchInput
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

# app
from app.tools.reranker import reranker




serp_api_key = settings.SERP_API_KEY



def search_google(query: str, api_key: str,city: str, state: str,country: str, num_results: int = 3) -> Dict[str, Any]:
    
    params = {
        "engine": "google",
        "q": query,
        "location": "{city}, {state}, {country}".format(city=city, state=state, country=country),
        "gl": "in",
        "hl": "en",
        "num": num_results,
        "api_key": api_key,
    }
    search = GoogleSearch(params)
    return search.get_dict()


def serp_results_to_documents(results: Dict[str, Any], tumor_type: str = "glioma") -> List[Document]:
    docs: List[Document] = []

    # Prioritize local_results if present
    local_results = results.get("local_results", {})
    places = local_results.get("places", [])
    if places:
        for place in places:
            title = place.get("title", "")
            rating = place.get("rating", "")
            reviews = place.get("reviews_original", "")
            description = place.get("description", "")
            place_type = place.get("type", "")
            phone = place.get("phone", "")
            address = place.get("address", "")
            hours = place.get("hours", "")
            gps = place.get("gps_coordinates", {})
            latitude = gps.get("latitude", "")
            longitude = gps.get("longitude", "")
            website = place.get("links", {}).get("website", "")
            directions = place.get("links", {}).get("directions", "")
            gse = place.get("place_id_search", None)

            page_content = (
                f"Title: {title}\n"
                f"Rating: {rating} {reviews}\n"
                f"Type: {place_type}\n"
                f"Phone: {phone}\n"
                f"Address: {address}\n"
                f"Hours: {hours}\n"
                f"Description: {description}\n"
                f"Latitude: {latitude}, Longitude: {longitude}\n"
                f"Website: {website}\n"
                f"Directions: {directions}"
            )

            doc = Document(
                page_content=page_content,
                metadata={
                    "tumor_type": tumor_type,
                    "section": None,
                    "subsection": None,
                    "source": gse,
                    "url_link": website,
                    "page": None,
                    "store_id": 2
                }
            )
            docs.append(doc)
        return docs

    # Fallback to organic_results
    organic_results = results.get("organic_results", [])
    for item in organic_results:
        snippet = item.get("snippet", "")
        source = item.get("source", None)
        url = item.get("link", None)

        if snippet:
            doc = Document(
                page_content=snippet,
                metadata={
                    "tumor_type": tumor_type,
                    "section": None,
                    "subsection": None,
                    "source": source,
                    "url_link": url,
                    "page": None,
                    "store_id": 2
                }
            )
            docs.append(doc)

    return docs


class SerpRetriever(BaseRetriever):
    api_key: str
    tumor_type: str = "glioma"
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = search_google(
            query=query,
            api_key=self.api_key,
            city=self.city or "",
            state=self.state or "",
            country=self.country or ""
        )
        docs = serp_results_to_documents(results, tumor_type=self.tumor_type)
        return docs






@tool("search_web", args_schema=SerpSearchInput)
async def search_web(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """Search the web for hospitals, doctors, costs, or recent/local brain tumor info not in the knowledge base."""
    # Safety check for missing config object
    config_dict = runtime.config or {}
    configurable = config_dict.get("configurable", {})
    
    # Extract keys safely from the configurable dictionary
    city = configurable.get("city") or config_dict.get("city")
    state = configurable.get("state") or config_dict.get("state")
    tumor_type = configurable.get("tumor_type") or config_dict.get("tumor_type")
    country = configurable.get("country") or config_dict.get("country")
    
    retriever = SerpRetriever(
        api_key=serp_api_key,
        tumor_type=tumor_type,
        city=city,
        state=state,
        country=country,
    )

    docs: List = await retriever.ainvoke(query)

    if not docs:
        return "No relevant results found on the web."


    reranked = await reranker.rerank(docs, query, top_n=4)

    if not reranked:
        return "No relevant results found after reranking."

    context_blocks = [
        f"[Context {i}] (Score: {score:.4f})\n{text}"
        for i, (text, score) in enumerate(reranked, 1)
    ]

    return "\n\n".join(context_blocks)
