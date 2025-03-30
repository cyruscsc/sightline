import pytest
from unittest.mock import Mock, patch
from app.paper_reader.arxiv_paper import ArXivPaper
from langchain.schema import Document


@pytest.fixture
def mock_arxiv_result():
    """Create a mock arXiv result with sample data."""
    mock_result = Mock()
    mock_result.entry_id = "2403.12345"
    mock_result.title = "Sample Paper Title"
    # Create mock authors with proper name attributes
    mock_author1 = Mock()
    mock_author1.name = "Author 1"
    mock_author2 = Mock()
    mock_author2.name = "Author 2"
    mock_result.authors = [mock_author1, mock_author2]
    mock_result.published = "2024-03-20"
    mock_result.categories = ["cs.AI", "cs.LG"]
    mock_result.summary = "This is a sample abstract."
    mock_result.doi = "10.1234/example.12345"
    mock_result.pdf_url = "https://arxiv.org/pdf/2403.12345.pdf"
    mock_result.content = (
        "This is the full paper content. " * 10
    )  # Create some content for splitting
    return mock_result


@pytest.fixture
def mock_arxiv_client(mock_arxiv_result):
    """Create a mock arXiv client."""
    with patch("app.paper_reader.arxiv_paper.arxiv.Client") as mock_client:
        mock_instance = Mock()
        mock_instance.results.return_value = iter([mock_arxiv_result])
        mock_client.return_value = mock_instance
        yield mock_client


class TestArXivPaper:
    def test_init_with_valid_abs_url(self, mock_arxiv_client):
        """Test initialization with a valid /abs/ URL."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)
        assert paper.url == url
        assert paper.arxiv_id == "2403.12345"

    def test_init_with_valid_pdf_url(self, mock_arxiv_client):
        """Test initialization with a valid /pdf/ URL."""
        url = "https://arxiv.org/pdf/2403.12345.pdf"
        paper = ArXivPaper(url)
        assert paper.url == url
        assert paper.arxiv_id == "2403.12345"

    def test_init_with_invalid_url(self):
        """Test initialization with an invalid URL."""
        with pytest.raises(ValueError, match="Not a valid arXiv URL"):
            ArXivPaper("https://example.com")

    def test_init_with_unsupported_url_format(self):
        """Test initialization with an unsupported URL format."""
        with pytest.raises(ValueError, match="Unsupported arXiv URL format"):
            ArXivPaper("https://arxiv.org/other/2403.12345")

    def test_paper_details(self, mock_arxiv_client):
        """Test paper details retrieval."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)
        details = paper.details

        assert details["arxiv_id"] == "2403.12345"
        assert details["title"] == "Sample Paper Title"
        assert details["authors"] == ["Author 1", "Author 2"]
        assert details["published"] == "2024-03-20"
        assert details["categories"] == ["cs.AI", "cs.LG"]
        assert details["abstract"] == "This is a sample abstract."
        assert details["doi"] == "10.1234/example.12345"
        assert details["pdf_url"] == "https://arxiv.org/pdf/2403.12345.pdf"

    def test_documents(self, mock_arxiv_client):
        """Test document splitting and creation."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)
        documents = paper.documents

        assert isinstance(documents, list)
        assert all(isinstance(doc, Document) for doc in documents)
        assert len(documents) > 0  # Should have at least one document

        # Check first document metadata
        first_doc = documents[0]
        assert first_doc.metadata["arxiv_id"] == "2403.12345"
        assert first_doc.metadata["title"] == "Sample Paper Title"
        assert first_doc.metadata["chunk_index"] == 0
        assert "total_chunks" in first_doc.metadata

    def test_property_getters(self, mock_arxiv_client):
        """Test all property getters."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)

        assert paper.title == "Sample Paper Title"
        assert paper.authors == ["Author 1", "Author 2"]
        assert paper.abstract == "This is a sample abstract."
        assert paper.pdf_url == "https://arxiv.org/pdf/2403.12345.pdf"

    def test_get_paper_data(self, mock_arxiv_client):
        """Test get_paper_data method."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)
        data = paper.get_paper_data()

        assert "details" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert all(isinstance(doc, Document) for doc in data["documents"])

    def test_immutable_properties(self, mock_arxiv_client):
        """Test that property getters return copies to prevent modification."""
        url = "https://arxiv.org/abs/2403.12345"
        paper = ArXivPaper(url)

        # Try to modify the returned copies
        details = paper.details
        details["title"] = "Modified Title"
        assert paper.title == "Sample Paper Title"  # Original should be unchanged

        authors = paper.authors
        authors.append("Author 3")
        assert len(paper.authors) == 2  # Original should be unchanged
