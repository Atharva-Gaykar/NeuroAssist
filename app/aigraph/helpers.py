from app.database.models import Patient
from langchain_core.messages import HumanMessage
from app.core.schemas import ChatRequest 

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