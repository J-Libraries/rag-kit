from langchain_core.chat_history import InMemoryChatMessageHistory


class ChatHistoryManager:
    def __init__(self):
        self._store: dict[str, InMemoryChatMessageHistory] = {}

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def clear_all(self) -> None:
        self._store.clear()