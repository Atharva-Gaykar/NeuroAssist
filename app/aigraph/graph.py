from langgraph.graph import StateGraph, START, END
from app.aigraph.state import NeuroAssistState
from app.nodes.research_node import research_node
from app.nodes.query_rewrite_node import query_rewrite_node
from app.nodes.query_safety_check import query_safety_check
from app.nodes.chat_node import chat_node
from app.database.connection import pool


graph_builder = StateGraph(NeuroAssistState)


# Creating nodes for graph
graph_builder.add_node("research_node", research_node)
graph_builder.add_node("query_rewrite_node", query_rewrite_node)
graph_builder.add_node("query_safety_check",query_safety_check)
graph_builder.add_node("chat_node", chat_node)



# connecting nodes
graph_builder.add_edge(START, "query_safety_check")
graph_builder.add_edge("query_safety_check", "query_rewrite_node")
graph_builder.add_edge("query_rewrite_node", "research_node")
graph_builder.add_edge("research_node", "chat_node")
graph_builder.add_edge("chat_node", END)

def get_compiled_graph(checkpointer):
    """Compiles and returns the runtime graph instance with active storage dependencies."""
    return graph_builder.compile(checkpointer=checkpointer)





