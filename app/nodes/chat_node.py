from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langmem.short_term import summarize_messages   # import may vary depending on your version
from app.agents.chat_agent  import chat_agent
from app.agents.chat_sumarising_agent import chat_sumarising_agent
from app.prompts.chat_agent_prompt import chat_agent_prompt
from app.prompts.message_summarising_agent_prompt import message_summarising_agent_prompt
from app.aigraph.state import NeuroAssistState


async def chat_node(state: NeuroAssistState, config: RunnableConfig):
    # Current user question is the latest message
    question = state["chat_agent_messages"][-1].content

    # Everything before that is conversation history
    chat_history = state["chat_agent_messages"][:-1]

    # ---------- Langmem SUMMARIZATION LOGIC ----------
    summarization_result =summarize_messages(
        chat_history,
        running_summary=state.get("chat_summary"),
        token_counter=chat_agent.get_num_tokens_from_messages,
        model=chat_sumarising_agent,
        max_tokens=300 ,
        max_tokens_before_summary=800,
        max_summary_tokens=128,
        final_prompt=message_summarising_agent_prompt
    )

    # These are the messages that should go into the prompt
    chat_history = summarization_result.messages
    # ---------------------------------------------

    # Pull latest research response
    research_messages = state.get("research_agent_messages", [])
    context = ""
    for msg in reversed(research_messages):
        if isinstance(msg, AIMessage) and msg.content:
            context = msg.content
            break

    prompt_messages = chat_agent_prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question,
    )

    response = await chat_agent.ainvoke(prompt_messages, config=config)

    return {
        "chat_agent_messages": [AIMessage(content=response.content)],
        "chat_summary": summarization_result.running_summary,
    }