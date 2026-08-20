
from langchain_groq import ChatGroq


chat_sumarising_agent=ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.1,
)