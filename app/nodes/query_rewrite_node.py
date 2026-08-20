from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from app.agents import query_rewrite_agent
from app.aigraph.state import NeuroAssistState
from app.prompts.query_rewrite_agent_prompt import query_rewrite_agent_system_prompt
from app.utils.output_parser import CleanQueryParser

query_rewrite_chain = query_rewrite_agent_system_prompt | query_rewrite_agent| CleanQueryParser() 

async def query_rewrite_node(state: NeuroAssistState, config: RunnableConfig):
    chat_history = state["chat_agent_messages"]

    if not chat_history:
        return {}

    current_question = state.get('cleaned_user_query',state["chat_agent_messages"][-1])
    prior_history = chat_history[-5:-1]

    # No prior turns — nothing to resolve against, skip reformulation
    if not prior_history:
        return {"rewritten_query": current_question}

    reformulated_question = await query_rewrite_chain.ainvoke(
        {
            "chat_history": prior_history,
            "input": current_question,
            "tumor_type": state["tumor_type"],
        },
        config=config,
    )

    return {"rewritten_query": reformulated_question}
