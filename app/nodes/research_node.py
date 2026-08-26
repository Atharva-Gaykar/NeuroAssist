from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from app.agents.research_agent import research_agent
from app.aigraph.state import NeuroAssistState

async def research_node(state: NeuroAssistState, config: RunnableConfig):
    recent_chat_history = state["chat_agent_messages"][-3:]
    
    # Guard against empty chat history to prevent IndexError
    if not recent_chat_history:
        return {"research_agent_messages": []}

    # Extract the latest query from the last message
    last_message = recent_chat_history[-1]

    query = state.get("rewritten_query", last_message.content)


    # Construct the input messages without duplicating the last message
    input_messages = [
        HumanMessage(content=f"Query: {query}\n Brain tumor type: {state['tumor_type']}")
    ] 
    
    current_configurable = config.get("configurable", {})

    sub_agent_config: RunnableConfig = {
        **config,
        "configurable": {
            **current_configurable,
            "city": state.get("city"),
            "state": state.get("state"),
            "country": state.get("country"),
            "tumor_type": state.get("tumor_type"),
        }
    }

    result = await research_agent.ainvoke(
        {"messages": input_messages},
        config=sub_agent_config
    )

    # tool-call trace via add_messages
    return {"research_agent_messages": result["messages"]}