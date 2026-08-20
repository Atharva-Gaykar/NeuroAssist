import os
import json
import pickle
import asyncio
from pathlib import Path
from typing import List
import httpx
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
import re
# app
from app.tools.embeddings import embeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from core.config import settings
from langchain.tools import ToolRuntime
# app
from app.tools.reranker import reranker
from app.prompts.tool_output_compresser_agent_prompt import prompt
from app.agents.tool_output_compresser_agent import tool_output_compresser_agent
from langchain_core.tools import tool

# app
from app.agents.research_agent import base_chat_llm
from app.tools.tools_input_schema import VectorSearchInput


# PATHS
BASE_DIR = Path(__file__).resolve().parent
TUMOR_CHUNK_FOLDER = BASE_DIR / "tumor_chunks"
BM25_PKL_PATH = BASE_DIR / "bm25.pkl"




# LOAD DOCUMENTS
def load_documents_from_folder(
    folder_path: Path
) -> List[Document]:

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    documents = []

    for file in folder_path.iterdir():

        if file.suffix != ".json":
            continue

        with open(file, "r", encoding="utf-8") as f:

            data = json.load(f)

            for item in data:

                documents.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"],
                    )
                )

    return documents


documents: List[Document] = load_documents_from_folder(
    TUMOR_CHUNK_FOLDER
)

if not documents:
    raise ValueError("No documents loaded from tumor_chunks")

print(f"Loaded {len(documents)} tumor chunks")



# PINECONE
pc = Pinecone(
    api_key=settings.PINECONE_API_KEY
)

INDEX_NAME = "neuroassist-data-fine-tuned-embeddings"

if INDEX_NAME not in pc.list_indexes().names():

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    print(f"Created index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)



# BM25


bm25_encoder = BM25Encoder()

if BM25_PKL_PATH.exists():

    print("Loading existing BM25 model...")

    with open(BM25_PKL_PATH, "rb") as f:
        bm25_encoder = pickle.load(f)

else:

    print("Fitting BM25 on tumor chunks...")

    bm25_encoder.fit(
        [doc.page_content for doc in documents]
    )

    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump(bm25_encoder, f)

    print("BM25 fitted and saved")



# ASYNC WRAPPER AROUND PINECONE HYBRID RETRIEVER

# PINECONE HYBRID RETRIEVER is synchronous
class AsyncPineconeHybridRetriever:

    def __init__(
        self,
        embeddings,
        sparse_encoder,
        index,
        top_k: int = 5,
        alpha: float = 0.5,
    ):

        self._retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=sparse_encoder,
            index=index,
            top_k=top_k,
            alpha=alpha,
        )

    async def ainvoke(
        self,
        query: str,
        search_kwargs: dict | None = None,
    ) -> List[Document]:

        config = {}

        if search_kwargs:
            config["search_kwargs"] = search_kwargs

        return await asyncio.to_thread(
            self._retriever.invoke,
            query,
            config,
        )

    def invoke(
        self,
        query: str,
        search_kwargs: dict | None = None,
    ) -> List[Document]:

        config = {}

        if search_kwargs:
            config["search_kwargs"] = search_kwargs

        return self._retriever.invoke(
            query,
            config,
        )


# RETRIEVER INSTANCE
pineretriever = AsyncPineconeHybridRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=7,
    alpha=0.35,
)


def strip_enrichment_header(text: str) -> str:
    """
    Removes the prepended '[Context: ...]' enrichment header from a chunk's
    text, added during ingestion to aid retrieval/reranking. The header has
    already done its job by this point (embedding + reranking), so it's
    stripped before the raw chunk is passed into the LLM's context window
    to save tokens without losing any actual medical content.
    """
    return re.sub(r"^\[Context:.*?\]\n?", "", text, flags=re.DOTALL).strip()



async def compress_tool_output(tool_output: str, query: str) -> str:
    final_prompt = prompt.format_messages(
        query=query,
        tool_output=tool_output,
    )

    result = await tool_output_compresser_agent.ainvoke(final_prompt)
    return result.content  # extract the string, not the AIMessage object




@tool("search_vector_db", args_schema=VectorSearchInput)
async def search_vector_db(query: str, runtime: ToolRuntime) -> str:
    """
    Search the internal medical knowledge base for established facts about
    brain tumor symptoms, causes, diagnosis, treatment, and prognosis.
    """
    config_dict = runtime.config or {}
    configurable = config_dict.get("configurable", {})

    tumor_type = configurable.get("tumor_type") or config_dict.get("tumor_type")

    if not tumor_type:
        return "Error: Missing tumor type configuration context."

    search_kwargs = {
        "k": 10,
        "filter": {"tumor_type": tumor_type.lower()},
    }

    docs = await pineretriever.ainvoke(query, search_kwargs=search_kwargs)

    if not docs:
        return "No relevant documents found in the knowledge base."

    reranked = await reranker.rerank(docs, query, top_n=4)

    if not reranked:
        return "No relevant documents found after reranking."

    filtered_docs = [
        (text, score) for text, score in reranked if score >= 0.73
    ]

    # If nothing cleared the threshold, fall back to the top unfiltered
    # results so the agent can see them and judge whether its query needs
    # refining — returning an empty string gives it nothing to work with.
    docs_to_format = filtered_docs if filtered_docs else reranked

    context_blocks = [
        f"[Context {i}] (Score: {score:.4f})\n{strip_enrichment_header(text)}"
        for i, (text, score) in enumerate(docs_to_format, 1)
    ]

    tool_output = "\n\n".join(context_blocks)

    if base_chat_llm.get_num_tokens(tool_output) > 550:
        tool_output = await compress_tool_output(tool_output, query)

    return tool_output








