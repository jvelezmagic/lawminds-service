from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory

from app.config import get_settings


def get_memory(session_id: str):
    settings = get_settings()

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        key_prefix="message_store:",
        ttl=60 * 60 * 24 * 7,  # 7 days
    )

    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history",
    )

    return memory
