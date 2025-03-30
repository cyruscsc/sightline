from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate


class PaperQA:
    def __init__(self, paper_data: dict):
        """
        Initialize a PaperQA instance with paper data and set up the RAG pipeline.

        Args:
            paper_data (Dict): Dictionary containing paper details and documents from ArXivPaper.get_paper_data()
        """
        self._paper_data = paper_data
        self._documents = paper_data["documents"]
        self._details = paper_data["details"]

        # Initialize embeddings and vector store
        self._embeddings = OpenAIEmbeddings()
        self._vectorstore = Chroma.from_documents(
            documents=self._documents,
            embedding=self._embeddings,
        )
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

        # Initialize the LLM
        self._llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Create the RAG chain
        self._qa_chain = self._create_qa_chain()

    def _format_context(self, context: list[Document]) -> str:
        """
        Format the context for the RAG chain.
        """
        return "\n".join([doc.page_content for doc in context])

    def _create_qa_chain(self) -> Runnable:
        """
        Create a RetrievalQA chain with custom prompt template.

        Returns:
            RetrievalQA: The configured QA chain
        """
        prompt_template = """You are an expert at answering questions about academic papers. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the question is not related to the paper, say that the question is not relevant to this paper.

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        rag_chain = (
            {
                "context": self._retriever | self._format_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        return rag_chain

    def ask_question(self, question: str) -> str:
        """
        Ask a question about the paper and get an answer using RAG.

        Args:
            question (str): The question to ask about the paper

        Returns:
            str: The answer to the question

        Raises:
            ValueError: If the question is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        result = self._qa_chain.invoke(question)

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
