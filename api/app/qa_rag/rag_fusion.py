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


class RAGFusion(BaseRAG):
    """
    RAG-Fusion system.
    """

    def __init__(self, retriever: VectorStoreRetriever, llm: ChatOpenAI):
        """
        Initialize the RAG-Fusion system.

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

    def _reciprocal_rank_fusion(
        self,
        results: list[list[Document]],
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Perform Reciprocal Rank Fusion on the results.

        Args:
            results (list[list[Document]]): List of lists of documents
        Returns:
            list[tuple[Document, float]]: List of tuples containing documents and their scores
        """
        # Initialize a dictionary to store the fused scores
        fused_scores = {}

        # Iterate through each list of documents
        for docs in results:
            # Iterate through each document in the list
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)

                # If the document is not already in the fused scores, add it
                # Otherwise, update the score using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)

        # Sort the fused scores in descending order
        reranked_results = [
            (loads(doc_str), score)
            for doc_str, score in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return reranked_results

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

    def _create_retrieval_chain(self) -> RunnableSerializable:
        """
        Create the retrieval chain for the RAG system.

        Returns:
            RunnableSerializable: The configured retrieval chain.
        """
        query_chain = self._create_query_chain()

        retrieval_chain = (
            query_chain
            | self._retriever.map()
            | RunnableLambda(self._reciprocal_rank_fusion)
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
