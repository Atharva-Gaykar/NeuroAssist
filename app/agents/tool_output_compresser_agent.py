from langchain_groq import ChatGroq


tool_output_compresser_agent=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.05,
)