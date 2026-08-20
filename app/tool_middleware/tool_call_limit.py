from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware


search_vector_db_tool_limiter = ToolCallLimitMiddleware(
    tool_name="search_vector_db",
    run_limit=4, 
    thread_limit=15,
    exit_behavior="continue"
)

web_tool_limiter = ToolCallLimitMiddleware(
    tool_name="search_web",
    run_limit=4, 
    thread_limit=15,
    exit_behavior="continue"
)
