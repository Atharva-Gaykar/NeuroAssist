from langchain.agents import create_agent
from langchain_groq import ChatGroq
from app.tools.websearchtool import search_web
from app.tools.vectordatabase import search_vector_db
from app.prompts.research_agent_prompt import research_agent_system_prompt
from app.tool_middleware.tool_call_limit import search_vector_db_tool_limiter, web_tool_limiter
from app.core.config import settings
import os


if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

base_chat_llm=ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.1,
)

research_agent=create_agent(
    model=base_chat_llm,
    tools=[search_vector_db, search_web],
    system_prompt=research_agent_system_prompt,
    middleware=[search_vector_db_tool_limiter, web_tool_limiter],
    debug=True
)