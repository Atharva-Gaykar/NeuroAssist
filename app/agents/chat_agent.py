
from langchain_groq import ChatGroq

chat_agent=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
)