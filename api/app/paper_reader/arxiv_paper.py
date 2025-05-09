import arxiv
from urllib.parse import urlparse
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os


class ArXivPaper:
    def __init__(self, url: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize an ArXivPaper instance with a given arXiv URL.

        Args:
            url (str): The arXiv paper URL (can be either /abs/ or /pdf/ format)

        Raises:
            ValueError: If the URL is not a valid arXiv URL or has an unsupported format
        """
        self._client = arxiv.Client()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize attributes
        self._url = url
        self._arxiv_id = self._extract_arxiv_id(url)
        self._paper = self._search_paper()
        self._details = self._get_paper_details()
        self._documents = self._get_paper_documents()

    def _extract_arxiv_id(self, url: str) -> str:
        """
        Extract arXiv ID from URL.

        Args:
            url (str): The arXiv paper URL

        Returns:
            str: The extracted arXiv ID

        Raises:
            ValueError: If the URL is not a valid arXiv URL or has an unsupported format
        """
        parsed_url = urlparse(url)

        if "arxiv.org" not in parsed_url.netloc:
            raise ValueError("Not a valid arXiv URL")

        # Handle different arXiv URL formats
        if "/abs/" in url:
            return parsed_url.path.split("/abs/")[-1]
        elif "/pdf/" in url:
            return parsed_url.path.split("/pdf/")[-1].replace(".pdf", "")
        else:
            raise ValueError("Unsupported arXiv URL format")

    def _search_paper(self) -> arxiv.Result:
        """
        Search for a paper by arXiv ID.

        Returns:
            arxiv.Result: The arXiv paper result object

        Raises:
            StopIteration: If no paper is found with the given arXiv ID
        """
        search = arxiv.Search(id_list=[self._arxiv_id])
        return next(self._client.results(search))

    def _get_paper_details(self) -> dict:
        """
        Get paper details from arXiv Result.

        Returns:
            Dict: Dictionary containing paper details including:
                - arxiv_id: The paper's arXiv ID
                - title: The paper's title
                - authors: List of author names
                - published: Publication date
                - categories: List of arXiv categories
                - abstract: Paper abstract
                - doi: DOI if available
                - pdf_url: URL to the PDF version
        """
        return {
            "arxiv_id": self._paper.entry_id,
            "title": self._paper.title,
            "authors": [author.name for author in self._paper.authors],
            "published": self._paper.published,
            "categories": self._paper.categories,
            "abstract": self._paper.summary,
            "doi": self._paper.doi,
            "pdf_url": self._paper.pdf_url,
        }

    def _get_paper_documents(self) -> list[Document]:
        """
        Get paper documents from arXiv Result.

        Returns:
            List[Document]: List of LangChain Document objects, each containing:
                - page_content: A chunk of the paper's content
        """
        # Download and get the paper content
        pdf_path = self._paper.download_pdf()
        reader = PdfReader(pdf_path)
        
        # Extract text from all pages
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"

        # Delete the PDF file after processing
        os.remove(pdf_path)

        # Split the content into chunks
        chunks = self._text_splitter.split_text(content)

        # Create documents for each chunk
        documents = [
            Document(page_content=chunk)
            for chunk in chunks
        ]

        return documents

    def get_paper_data(self) -> dict:
        """
        Get paper details and documents in the original format.

        Returns:
            dict: Dictionary containing:
                - details: Paper metadata and information
                - documents: List of LangChain Document objects
        """
        return {
            "details": self._details,
            "documents": self._documents,
        }

    # Getters for private attributes
    @property
    def url(self) -> str:
        """
        Get the original arXiv URL.

        Returns:
            str: The arXiv paper URL
        """
        return self._url

    @property
    def arxiv_id(self) -> str:
        """
        Get the arXiv ID.

        Returns:
            str: The paper's arXiv ID
        """
        return self._arxiv_id

    @property
    def details(self) -> dict:
        """
        Get the paper details.

        Returns:
            Dict: Dictionary containing paper metadata and information
        """
        return self._details.copy()  # Return a copy to prevent modification

    @property
    def documents(self) -> list[Document]:
        """
        Get the paper documents.

        Returns:
            List[Document]: List of LangChain Document objects
        """
        return self._documents.copy()  # Return a copy to prevent modification

    @property
    def title(self) -> str:
        """
        Get the paper title.

        Returns:
            str: The paper's title
        """
        return self._details["title"]

    @property
    def authors(self) -> list[str]:
        """
        Get the paper authors.

        Returns:
            List[str]: List of author names
        """
        return self._details["authors"].copy()  # Return a copy to prevent modification

    @property
    def abstract(self) -> str:
        """
        Get the paper abstract.

        Returns:
            str: The paper's abstract
        """
        return self._details["abstract"]

    @property
    def pdf_url(self) -> str:
        """
        Get the PDF URL.

        Returns:
            str: URL to the paper's PDF version
        """
        return self._details["pdf_url"]
