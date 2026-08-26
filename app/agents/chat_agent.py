
from langchain_groq import ChatGroq
from app.core.config import settings
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

chat_agent=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)