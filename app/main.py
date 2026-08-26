from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException , Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
import uvicorn
import logging
import traceback
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi.sse import EventSourceResponse
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
import os
import markdown
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import SessionLocal, get_db
from app.core.auth import verify_token, create_access_token, hash_password, verify_password,get_current_patient
from app.database.crud import (
    get_patient_by_email,
    create_patient,
)
import uuid
from app.database.models import Patient
from pathlib import Path
from datetime import datetime
from app.core.config import settings
import cloudinary
import cloudinary.uploader
import io
from app.CNN.cnnapi import process_tumor_detection
from app.database.connection import pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from collections.abc import AsyncIterable
from app.aigraph.graph import get_compiled_graph




print("all imports done")

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)



#Used for local testing
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

# To configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.mount("/static", StaticFiles(directory="static"), name="static")




class ChatRequest(BaseModel):
    query: str

class TokenResponse(BaseModel):
    success: bool
    token: str
    message: str



def get_graph_input(
    patient: Patient,
    request: ChatRequest
):
    return {
        "tumor_type": patient.tumor_type,
        "city": patient.city,
        "state": patient.state,
        "country": patient.country,

        "chat_agent_messages": [
            HumanMessage(
                content=request.message
            )
        ],

        "research_agent_messages": [],
    }

async def initialize_patient_graph_state(graph, patient):
    config = {
        "configurable": {
            "user_id": patient.id,
            "thread_id": str(patient.user_chat_thread),
        }
    }

    await graph.aupdate_state(
        config,
        {
            "tumor_type": patient.tumor_type,
            "city": patient.city,
            "state": patient.state,
            "country": patient.country,
            "chat_agent_messages": [],
            "research_agent_messages": [],
        },
    )

    return config
# User registration endpoint


@app.post("/api/register")
async def register_patient(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    country: str = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Registers a new patient.

    Uses Multipart Form Data because an MRI image is uploaded.
    """

    try:
        # ---------------------------------------------------------
        # 1. Check whether email already exists
        # ---------------------------------------------------------
        existing_patient = await get_patient_by_email(db, email)

        if existing_patient:
            raise HTTPException(
                status_code=400,
                detail="Email already registered",
            )

        # ---------------------------------------------------------
        # 2. Read MRI image once
        # ---------------------------------------------------------
        content = await image.read()

        # ---------------------------------------------------------
        # 3. Upload MRI to Cloudinary
        # ---------------------------------------------------------
        upload_result = await asyncio.to_thread(
            cloudinary.uploader.upload,
            io.BytesIO(content),
            folder="neuroassist/mri_scans",
            resource_type="image",
        )

        image_url = upload_result["secure_url"]
        image_public_id = upload_result["public_id"]

        # ---------------------------------------------------------
        # 4. Run tumor detection
        # ---------------------------------------------------------
        tumor_result = await process_tumor_detection(content)
        tumor_type = tumor_result["tumor_type"]

        # ---------------------------------------------------------
        # 5. Create patient
        #
        # create_patient() should generate user_chat_thread
        # using uuid.uuid4() and persist it in PostgreSQL.
        # ---------------------------------------------------------
        patient_data = {
            "username": username,
            "email": email,
            "password": hash_password(password[:72]),
            "city": city,
            "state": state,
            "country": country,
        }

        user_chat_thread = uuid.uuid4()

        new_patient = await create_patient(
            db=db,
            patient_data=patient_data,
            tumor_type=tumor_type,
            image_url=image_url,
            user_chat_thread=user_chat_thread,
            image_public_id=image_public_id,
        )

        # ---------------------------------------------------------
        # 6. Initialize LangGraph persistent state
        #
        # This DOES NOT invoke the graph.
        # It only stores the initial state for this thread.
        # ---------------------------------------------------------
        graph = app.state.graph

        await initialize_patient_graph_state(
            graph=graph,
            patient=new_patient,
        )

        # ---------------------------------------------------------
        # 7. Create authentication token
        # ---------------------------------------------------------
        token = create_access_token(
            {
                "patient_id": new_patient.id,
            }
        )

        # ---------------------------------------------------------
        # 8. Return registration response
        # ---------------------------------------------------------
        return {
            "success": True,
            "token": token,
            "patient": {
                "id": new_patient.id,
                "username": new_patient.username,
                "tumor_type": new_patient.tumor_type,
                "image_url": new_patient.mri_image_url,
                "user_chat_thread": str(
                    new_patient.user_chat_thread
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        print(f"[REGISTER ERROR] {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )




# sign_in Endpoint
@app.post("/api/signin")
async def sign_in(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    user = await get_patient_by_email(db, email)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(password, user.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token({
        "patient_id": user.id,
    })

    return {
        "success": True,
        "token": token,
        "patient": {
            "id": user.id,
            "username": user.username,
            "tumor_type": user.tumor_type
        }
    }




# To gt patient chat history
@app.get("/api/patient/info")
async def get_patient_profile(patient: Patient = Depends(get_current_patient)):
    """Returns profile for the currently logged-in user"""
    return {
        "success": True,
        "patient": {
            "id": patient.id,
            "username": patient.username,
            "email": patient.email,
            "city": patient.city,
            "tumor_type": patient.tumor_type
        }
    }






@app.post(
    "/api/chat/message",
    response_class=EventSourceResponse
)
async def chat_message(
    chat_request: ChatRequest,
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
) -> AsyncIterable[ServerSentEvent]:

    graph = getattr(
        request.app.state,
        "graph",
        None
    )

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Chat service is not available right now."
        )

    # Validate message
    if (
        not chat_request.message
        or not chat_request.message.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Get user's persistent chat thread
    patient_chat_thread = patient.user_chat_thread

    # Build graph input
    graph_input = get_graph_input(
        patient,
        chat_request
    )

    # LangGraph configuration
    config = {
        "configurable": {
            "user_id": patient.id,
            "thread_id": patient_chat_thread,
        }
    }

    try:

        async for event in graph.astream_events(
            graph_input,
            config=config,
            version="v2"
        ):

            # Stop processing if client disconnects
            if await request.is_disconnected():

                logger.info(
                    "Client disconnected for user_id=%s, thread_id=%s",
                    patient.id,
                    patient_chat_thread,
                )

                break

            kind = event.get("event")

            # Only stream tokens generated by chat_node
            if (
                kind == "on_chat_model_stream"
                and event.get(
                    "metadata",
                    {}
                ).get("langgraph_node") == "chat_node"
            ):

                chunk = event.get(
                    "data",
                    {}
                ).get("chunk")

                if chunk and chunk.content:

                    yield ServerSentEvent(
                        event="token",
                        data={
                            "text": chunk.content
                        }
                    )

        # Send completion event
        yield ServerSentEvent(
            event="end",
            data={
                "status": "completed"
            }
        )

    except asyncio.CancelledError:

        logger.info(
            "Client disconnected during streaming "
            "for user_id=%s, thread_id=%s",
            patient.id,
            patient_chat_thread,
        )

        raise

    except Exception:

        logger.exception(
            "Streaming chat pipeline failed for user_id=%s, thread_id=%s",
            patient.id,
            patient_chat_thread,
        )

        yield ServerSentEvent(
            event="error",
            data={
                "message": "Error generating response"
            }
        )
         





             

         
















    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise HTTPException(status_code=500, detail="Error generating response")




#used for local testing, not needed in production or HF Spaces since they handle server startup
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f" API Running on http://127.0.0.1:8500 | Device: {device}")
#     uvicorn.run(app, host="127.0.0.1", port=8500)




