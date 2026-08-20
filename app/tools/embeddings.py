from langchain_core.embeddings import Embeddings
import httpx
import asyncio
from langchain_core.embeddings import Embeddings
import httpx

class MedicalRemoteEmbeddings(Embeddings):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        
        # Shared connection pools for speed. No rebuilding on every call!
        self._timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        
        self._async_client = httpx.AsyncClient(timeout=self._timeout)
        self._sync_client = httpx.Client(timeout=self._timeout)

    # this part is calling the embed_query endpoint of the rag medical embeddings service asynchronously
    # rag_medical_embedding is  a asynchrounus service.
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._async_client.post(
            f"{self.endpoint}/embed_docs",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    

    # this part is calling the embed_query endpoint of the rag medical embeddings service asynchronously
    # rag_medical_embedding is  a asynchrounus service.

    async def aembed_query(self, text: str) -> list[float]:
        response = await self._async_client.post(
            f"{self.endpoint}/embed_query",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    # --- Sync Methods (Used by Pinecone inside your asyncio.to_thread) ---
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._sync_client.post(
            f"{self.endpoint}/embed_docs",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        response = self._sync_client.post(
            f"{self.endpoint}/embed_query",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    # Clean up both clients on app shutdown
    async def aclose(self):
        await self._async_client.aclose()
        self._sync_client.close()



embeddings = MedicalRemoteEmbeddings(
    endpoint="https://gaykar-rag-medical-embeddings.hf.space"
)
