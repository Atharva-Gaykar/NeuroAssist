from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

query_rewrite_agent_system_prompt = """
You are helping answer questions about brain tumors, specifically '{tumor_type}'.
Given a chat history and the latest user question which might reference context
in the chat history, formulate a standalone question which can be understood
without the chat history and used by the research agent. Do NOT answer the question.
"""

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", query_rewrite_agent_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])