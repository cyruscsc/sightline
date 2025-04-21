from .base_rag import BaseRAG
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from textwrap import dedent


class MultiQuery(BaseRAG):
    """
    MultiQuery RAG system.
    """

    def __init__(self, retriever: VectorStoreRetriever, llm: ChatOpenAI):
        """
        Initialize the MultiQuery RAG system.

        Args:
            retriever (VectorStoreRetriever): The vector store retriever.
        """
        self._retriever = retriever
        self._llm = llm
        self._query_prompt_template = dedent(
            """\
            You are an AI language model assistant.
            Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
            Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )
        self._generation_prompt_template = dedent(
            """\
            You are an expert at answering questions about academic papers.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If the question is not related to the paper, say that the question is not relevant to this paper.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )

    def _get_unique_union(self, documents: list[list[Document]]) -> list[Document]:
        """
        Get unique documents from a list of lists of documents.

        Args:
            documents (list[list[Document]]): List of lists of documents
        Returns:
            list[Document]: List of unique documents
        """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def _join_documents(self, documents: list[Document]) -> str:
        """
        Join the content of documents into a single string.

        Args:
            documents (list[Document]): List of documents
        Returns:
            str: Joined content of the documents
        """
        return "\n".join([doc.page_content for doc in documents])

    def _create_query_chain(self) -> RunnableSerializable:
        """
        Create the query chain for the RAG system.

        Returns:
            RunnableSerializable: The configured query chain.
        """
        query_chain = (
            ChatPromptTemplate.from_template(self._query_prompt_template)
            | self._llm
            | StrOutputParser()
            | RunnableLambda(lambda x: x.split("\n"))
        )

        return query_chain

    def _create_retrieval_chain(self):
        """
        Create the retrieval chain for the RAG system.

        Returns:
            RunnableSerializable: The configured retrieval chain.
        """
        query_chain = self._create_query_chain()

        retrieval_chain = (
            query_chain
            | self._retriever.map()
            | RunnableLambda(self._get_unique_union)
            | RunnableLambda(self._join_documents)
        )

        return retrieval_chain

    def _create_rag_chain(self) -> RunnableSerializable:
        """
        Create the RAG chain for the RAG system.

        Returns:
            RunnableSerializable: The configured RAG chain.
        """
        retrieval_chain = self._create_retrieval_chain()

        rag_chain = (
            {
                "context": retrieval_chain,
                "question": itemgetter("question"),
            }
            | ChatPromptTemplate.from_template(self._generation_prompt_template)
            | self._llm
            | StrOutputParser()
        )

        return rag_chain

    async def generate(self, question: str) -> str:
        """
        Generate the answer for the given question.

        Args:
            question (str): The question to answer.
        Returns:
            str: The generated answer.
        """
        rag_chain = self._create_rag_chain()

        result = await rag_chain.ainvoke({"question": question})

        return result
