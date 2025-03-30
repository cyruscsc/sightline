import pytest
from app.paper_reader.arxiv_paper import ArXivPaper
from langchain.schema import Document


@pytest.fixture
def real_arxiv_result():
    """Create a real arXiv result with the actual paper data."""
    paper = ArXivPaper("https://arxiv.org/abs/1706.03762")
    return paper._paper


@pytest.fixture
def arxiv_paper():
    """Create an ArXivPaper instance with the actual paper URL."""
    return ArXivPaper("https://arxiv.org/abs/1706.03762")


class TestArXivPaper:
    def test_init_with_valid_abs_url(self):
        """Test initialization with a valid /abs/ URL."""
        url = "https://arxiv.org/abs/1706.03762"
        paper = ArXivPaper(url)
        assert paper.url == url
        assert paper.arxiv_id == "1706.03762"

    def test_init_with_valid_pdf_url(self):
        """Test initialization with a valid /pdf/ URL."""
        url = "https://arxiv.org/pdf/1706.03762.pdf"
        paper = ArXivPaper(url)
        assert paper.url == url
        assert paper.arxiv_id == "1706.03762"

    def test_init_with_invalid_url(self):
        """Test initialization with an invalid URL."""
        with pytest.raises(ValueError, match="Not a valid arXiv URL"):
            ArXivPaper("https://example.com")

    def test_init_with_unsupported_url_format(self):
        """Test initialization with an unsupported URL format."""
        with pytest.raises(ValueError, match="Unsupported arXiv URL format"):
            ArXivPaper("https://arxiv.org/other/1706.03762")

    def test_paper_details(self, arxiv_paper):
        """Test paper details retrieval."""
        details = arxiv_paper.details

        # Check arxiv_id with version
        assert "1706.03762" in details["arxiv_id"]
        assert details["title"] == "Attention Is All You Need"
        assert "Ashish Vaswani" in details["authors"]  # First author
        assert details["published"] is not None
        assert "cs.CL" in details["categories"]  # Paper category
        assert details["abstract"] is not None
        # DOI is optional, so we don't assert it
        # Check pdf_url with version
        assert "1706.03762" in details["pdf_url"]
        assert "pdf" in details["pdf_url"].lower()

    def test_documents(self, arxiv_paper):
        """Test document splitting and creation."""
        documents = arxiv_paper.documents

        assert isinstance(documents, list)
        assert all(isinstance(doc, Document) for doc in documents)
        assert len(documents) > 0  # Should have at least one document

        # Check first document metadata
        first_doc = documents[0]
        assert "1706.03762" in first_doc.metadata["arxiv_id"]
        assert first_doc.metadata["title"] == "Attention Is All You Need"
        assert first_doc.metadata["chunk_index"] == 0
        assert "total_chunks" in first_doc.metadata

    def test_property_getters(self, arxiv_paper):
        """Test all property getters."""
        assert arxiv_paper.title == "Attention Is All You Need"
        assert "Ashish Vaswani" in arxiv_paper.authors  # First author
        assert arxiv_paper.abstract is not None
        # Check pdf_url with version
        assert "1706.03762" in arxiv_paper.pdf_url
        assert "pdf" in arxiv_paper.pdf_url.lower()

    def test_get_paper_data(self, arxiv_paper):
        """Test get_paper_data method."""
        data = arxiv_paper.get_paper_data()

        assert "details" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert all(isinstance(doc, Document) for doc in data["documents"])

    def test_immutable_properties(self, arxiv_paper):
        """Test that property getters return copies to prevent modification."""
        # Try to modify the returned copies
        details = arxiv_paper.details
        details["title"] = "Modified Title"
        assert arxiv_paper.title == "Attention Is All You Need"  # Original should be unchanged

        authors = arxiv_paper.authors
        authors.append("Author 3")
        assert "Author 3" not in arxiv_paper.authors  # Original should be unchanged
