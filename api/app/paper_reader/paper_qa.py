from app.qa_rag import SimpleRAG, MultiQuery, RAGFusion, HyDE
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import tempfile
import shutil


class PaperQA:
    def __init__(self, paper_data: dict, strategy: str = "simple"):
        """
        Initialize a PaperQA instance with paper data and set up the RAG pipeline.

        Args:
            paper_data (Dict): Dictionary containing paper details and documents from ArXivPaper.get_paper_data()
        """
        self._details = paper_data["details"]
        self._documents = paper_data["documents"]
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
                case "multi-query":
                    rag = MultiQuery(retriever, self._llm)
                case "rag-fusion":
                    rag = RAGFusion(retriever, self._llm)
                case "hyde":
                    rag = HyDE(retriever, self._llm)
                case _:
                    rag = SimpleRAG(retriever, self._llm)

            result = await rag.generate(question)

        finally:
            # Clean up by deleting the temporary directory
            shutil.rmtree(temp_dir)

        # TODO: Add confidence and source sections

        return {"answer": result}
