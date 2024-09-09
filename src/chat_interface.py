from abc import ABC, abstractmethod


class ChatInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def count_tokens(self, messages: list) -> int:
        raise NotImplementedError

    @abstractmethod
    def chat_parallel(self, message_list: list) -> list:
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list, stream: bool) -> list:
        raise NotImplementedError
