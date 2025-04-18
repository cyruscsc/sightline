from operator import itemgetter
from langchain.load import dumps, loads
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from textwrap import dedent
import tempfile
import shutil


class PaperQA:
    def __init__(self, paper_data: dict, strategy: str = "simple"):
        """
        Initialize a PaperQA instance with paper data and set up the RAG pipeline.

        Args:
            paper_data (Dict): Dictionary containing paper details and documents from ArXivPaper.get_paper_data()
        """
        self._paper_data = paper_data
        self._documents = paper_data["documents"]
        self._details = paper_data["details"]
        self._embeddings = OpenAIEmbeddings()
        self._llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self._strategy = strategy

    def _create_retriever(self, temp_dir: str) -> VectorStoreRetriever:
        """
        Create a retriever for the vector store.

        Returns:
            VectorStoreRetriever: The configured retriever
        """
        vectorstore = Chroma.from_documents(
            documents=self._documents,
            embedding=self._embeddings,
            persist_directory=temp_dir,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    def _get_unique_union(self, doccuments: list[list[Document]]) -> list[Document]:
        """
        Get unique documents from a list of lists of documents.

        Args:
            doccuments (list[list[Document]]): List of lists of documents
        Returns:
            list[Document]: List of unique documents
        """
        flattened_docs = [dumps(doc) for sublist in doccuments for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def _reciprocal_rank_fusion(
        results: list[list[Document]], k: int = 60
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

    def _create_simple_retrieval_chain(
        self, retriever: VectorStoreRetriever
    ) -> RunnableSerializable:
        """
        Create a RAG retrieval chain with a single query.

        Args:
            retriever (VectorStoreRetriever): The vector store retriever
        Returns:
            RunnableSerializable: The configured RAG retrieval chain
        """
        retrieval_chain = (
            itemgetter("question")
            | retriever
            | RunnableLambda(lambda x: "\n".join([doc.page_content for doc in x]))
        )

        return retrieval_chain

    def _create_multi_query_chain(self) -> RunnableSerializable:
        """
        Create a multi-query chain for generating multiple perspectives on the user question.

        Returns:
            RunnableSerializable: The configured multi-query chain
        """
        prompt_template = dedent(
            """\
            You are an AI language model assistant.
            Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
            Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )
        prompt_perspectives = ChatPromptTemplate.from_template(prompt_template)

        query_chain = (
            prompt_perspectives
            | self._llm
            | StrOutputParser()
            | RunnableLambda(lambda x: x.split("\n"))
        )

        return query_chain

    def _create_multi_query_retrieval_chain(
        self, retriever: VectorStoreRetriever
    ) -> RunnableSerializable:
        """
        Create a RAG retrieval chain with multi-query.

        Args:
            retriever (VectorStoreRetriever): The vector store retriever
        Returns:
            RunnableSerializable: The configured multi-query retrieval chain
        """
        query_chain = self._create_multi_query_chain()

        retrieval_chain = (
            query_chain
            | retriever.map()
            | RunnableLambda(self._get_unique_union)
            | RunnableLambda(lambda x: "\n".join([doc.page_content for doc in x]))
        )

        return retrieval_chain

    def _create_rag_fusion_retrieval_chain(
        self, retriever: VectorStoreRetriever
    ) -> RunnableSerializable:
        """
        Create a RAG retrieval chain with RAG-Fusion.

        Args:
            retriever (VectorStoreRetriever): The vector store retriever
        Returns:
            RunnableSerializable: The configured RAG-Fusion retrieval chain
        """
        query_chain = self._create_multi_query_chain()

        retrieval_chain = (
            query_chain
            | retriever.map()
            | RunnableLambda(self._reciprocal_rank_fusion)
            | RunnableLambda(lambda x: "\n".join([doc.page_content for doc in x]))
        )

        return retrieval_chain

    def _create_hyde_retrieval_chain(
        self, retriever: VectorStoreRetriever
    ) -> RunnableSerializable:
        """
        Create a RAG retrieval chain with HyDE.

        Args:
            retriever (VectorStoreRetriever): The vector store retriever
        Returns:
            RunnableSerializable: The configured HyDE retrieval chain
        """
        prompt_template = dedent(
            """\
            You are an AI academic research assistant.
            Please write an academic passage to answer the following question.
            Question: {question}"""
        )
        prompt_hyde = ChatPromptTemplate.from_template(prompt_template)

        query_chain = prompt_hyde | self._llm | StrOutputParser()

        retrieval_chain = (
            query_chain
            | retriever
            | RunnableLambda(lambda x: "\n".join([doc.page_content for doc in x]))
        )

        return retrieval_chain

    def _create_rag_chain(
        self, retrieval_chain: RunnableSerializable
    ) -> RunnableSerializable:
        """
        Create a RAG chain with custom prompt template.

        Args:
            retrieval_chain (RunnableSerializable): The retrieval chain to use

        Returns:
            RunnableSerializable: The configured RAG chain
        """
        prompt_template = dedent(
            """
            You are an expert at answering questions about academic papers.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If the question is not related to the paper, say that the question is not relevant to this paper.

            Context:
            {context}

            Question: {question}

            Answer:"""
        )

        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = (
            {
                "context": retrieval_chain,
                "question": itemgetter("question"),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        return rag_chain

    async def ask_question(self, question: str) -> dict:
        """
        Ask a question about the paper and get an answer using RAG.

        Args:
            question (str): The question to ask about the paper

        Returns:
            dict: Dictionary with the answer to the question

        Raises:
            ValueError: If the question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Create a temporary directory for Chroma
        temp_dir = tempfile.mkdtemp()

        try:
            retriever = self._create_retriever(temp_dir)

            match self._strategy:
                case "multi_query":
                    retrieval_chain = self._create_multi_query_retrieval_chain(
                        retriever
                    )
                case "rag_fusion":
                    retrieval_chain = self._create_rag_fusion_retrieval_chain(retriever)
                case "hyde":
                    retrieval_chain = self._create_hyde_retrieval_chain(retriever)
                case _:
                    retrieval_chain = self._create_simple_retrieval_chain(retriever)

            rag_chain = self._create_rag_chain(retrieval_chain)

            result = await rag_chain.ainvoke({"question": question})

        finally:
            # Clean up by deleting the temporary directory
            shutil.rmtree(temp_dir)

        # TODO: Add confidence and source sections

        return {"answer": result}

    @property
    def paper_details(self) -> dict:
        """
        Get the paper details.

        Returns:
            Dict: Dictionary containing paper metadata and information
        """
        return self._details.copy()

    @property
    def documents(self) -> list[Document]:
        """
        Get the paper documents.

        Returns:
            List[Document]: List of LangChain Document objects
        """
        return self._documents.copy()
