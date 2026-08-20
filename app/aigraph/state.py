from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langmem.short_term import RunningSummary

class NeuroAssistState(TypedDict):
    tumor_type: str
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    chat_summary: RunningSummary|None
    rewritten_query: Optional[str]
    cleaned_user_query:Optional[str]
    chat_agent_messages: Annotated[List[BaseMessage], add_messages]
    research_agent_messages: Annotated[List[BaseMessage], add_messages]






# Helper function  to initialize the graph state
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