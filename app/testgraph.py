import asyncio
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.aigraph.graph import get_compiled_graph
from app.database.connection import pool

async def run_local_test():
    # 1. Open the background network sockets manually
    print("Opening database connection pool...")
    await pool.open()
    
    # 2. Setup your checkpointer and run database migrations
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    
    # 3. Compile the graph ONCE using the active checkpointer
    graph = get_compiled_graph(checkpointer=checkpointer)
    print("Graph compiled successfully with checkpointer persistence!")

    # 4. Define your testing configuration
    config = {
        "configurable": {
            "user_id": "test_user_12345",
            "thread_id": str(uuid.uuid4()),
        }
    }

    # 5. Define your initial execution state
    state = {
        "tumor_type": "glioma",
        "city": "Pune",
        "state": "Maharashtra",
        "country": "India",
        "chat_agent_messages": [
            HumanMessage(content="Hi i am Atharva Gaykar, can you tell me what treatments are given for this tumor?")
        ],
        "research_agent_messages": [],
    }

    # 6. Execute the graph asynchronously
    print("Invoking graph workflow...")
    result_1 = await graph.ainvoke(state, config=config)

    # 7. Print the results
    print("\n========== TURN 1 ==========")
    print("--- RESEARCH AGENT MESSAGES ---")
    for msg in result_1.get("research_agent_messages", []):
        print(f"[{msg.type}] {msg.content}\n")

    print("--- CHAT AGENT MESSAGES ---")
    for msg in result_1.get("chat_agent_messages", []):
        print(f"[{msg.type}] {msg.content}\n")

    # 8. Crucial step: Safely drain and disconnect the pool channels
    print("Closing database connection pool cleanly...")
    await pool.close()

# --- Execution Section ---
if __name__ == "__main__":
    # If running as a standard terminal Python script (.py file):
    asyncio.run(run_local_test())

