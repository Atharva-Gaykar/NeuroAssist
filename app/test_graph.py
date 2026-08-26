import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException , Depends
from fastapi import HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.aigraph.graph import get_compiled_graph
from app.database.connection import pool
from collections.abc import AsyncIterable
import uvicorn
import sys
import json
import asyncio

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Open the database connection pool
    await pool.open()

    # Initialize the graph with a checkpointer
    checkpointer = AsyncPostgresSaver(pool)
    app.state.graph = get_compiled_graph(checkpointer)

    yield

    # Close the database connection pool
    await pool.close()


app = FastAPI(title="NeuroAssist", lifespan=lifespan)

# Schema to manually provide all test parameters
class ManualTestChatRequest(BaseModel):
    message: str
    user_id: str = "test_user_123"
    thread_id: str = "test_thread_abc"
    tumor_type: Optional[str] = "Glioblastoma"
    city: Optional[str] = "New York"
    state: Optional[str] = "NY"
    country: Optional[str] = "USA"



@app.post(
    "/api/chat/message/stream",
    response_class=EventSourceResponse
)
async def chat_message_stream(
    request: ManualTestChatRequest,
    fastapi_request: Request,
) -> AsyncIterable[ServerSentEvent]:

    graph = getattr(
        fastapi_request.app.state,
        "graph",
        None
    )

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Chat service is not available right now."
        )

    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    config = {
        "configurable": {
            "user_id": request.user_id,
            "thread_id": request.thread_id,
        }
    }

    graph_input = {
        "tumor_type": request.tumor_type,
        "city": request.city,
        "state": request.state,
        "country": request.country,
        "chat_agent_messages": [
            HumanMessage(content=request.message)
        ],
        "research_agent_messages": [],
    }

    try:

        async for event in graph.astream_events(
            graph_input,
            config=config,
            version="v2"
        ):

            if await fastapi_request.is_disconnected():

                logger.info(
                    "Client disconnected for user_id=%s",
                    request.user_id
                )

                break

            kind = event.get("event")

            if (
                kind == "on_chat_model_stream"
                and event.get(
                    "metadata",
                    {}
                ).get("langgraph_node") == "chat_node"
            ):

                chunk = event["data"].get("chunk")

                if chunk and chunk.content:

                    yield ServerSentEvent(
                        event="token",
                        data={
                            "text": chunk.content
                        }
                    )

        yield ServerSentEvent(
            event="end",
            data={
                "status": "completed"
            }
        )

    except asyncio.CancelledError:

        logger.info(
            "Client disconnected during streaming"
        )

        raise

    except Exception:

        logger.exception(
            "Streaming chat pipeline failed for user_id=%s, thread_id=%s",
            request.user_id,
            request.thread_id,
        )

        yield ServerSentEvent(
            event="error",
            data={
                "message": "Error generating response"
            }
        )


if __name__ == "__main__":
    import sys
    import asyncio

    # Apply the event loop policy fix for Windows execution
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Point Uvicorn to your actual module path (app.testgraph) instead of "main"
    uvicorn.run("app.test_graph:app", host="127.0.0.1", port=8500)

