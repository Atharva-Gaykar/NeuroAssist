from langchain_groq import ChatGroq

from app.core.config import settings
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

chat_sumarising_agent=ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.1,
)