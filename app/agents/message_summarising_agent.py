from langchain_groq import ChatGroq


message_summarising_agent=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
)