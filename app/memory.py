from typing import Any, Dict

from langchain.memory import ConversationTokenBufferMemory, RedisChatMessageHistory
from langchain.schema import BaseMessage
from langchain.schema.messages import BaseMessage, get_buffer_string


class ConversationTokenBufferMemoryFromDatabase(ConversationTokenBufferMemory):
    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    def prune_buffer(self):
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory: list[BaseMessage] = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.chat_memory.messages = buffer

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is True."""
        self.prune_buffer()
        return self.chat_memory.messages

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is False."""
        self.prune_buffer()
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super().load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)


from langchain.schema.language_model import BaseLanguageModel

from app.config import get_settings


def get_memory(
    session_id: str,
    llm: BaseLanguageModel,
    return_messages: bool = True,
):
    settings = get_settings()

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        key_prefix="message_store:",
        ttl=60 * 60 * 24 * 7,  # 7 days
    )

    memory = ConversationTokenBufferMemory(
        chat_memory=chat_history,
        return_messages=return_messages,
        memory_key="chat_history",
        llm=llm,
    )

    return memory
