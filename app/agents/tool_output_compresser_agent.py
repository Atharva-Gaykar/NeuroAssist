from langchain_groq import ChatGroq
from app.core.config import settings
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY


tool_output_compresser_agent=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.05,
)