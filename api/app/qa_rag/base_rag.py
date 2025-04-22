from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever


class BaseRAG(ABC):
    """
    Base class for Retrieval-Augmented Generation (RAG) systems.
    This class defines the interface for RAG systems and provides a common
    structure for implementing different RAG models.
    """

    @abstractmethod
    def _create_query_chain(self) -> RunnableSerializable:
        pass

    @abstractmethod
    def _create_retrieval_chain(self) -> RunnableSerializable:
        pass

    @abstractmethod
    def _create_rag_chain(self) -> RunnableSerializable:
        pass

    @abstractmethod
    async def generate(self, question: str) -> str:
        pass
