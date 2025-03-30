import pytest
from langchain.schema import Document
from app.paper_reader.paper_summarizer import PaperSummarizer, PaperSummary
from app.paper_reader.arxiv_paper import ArXivPaper
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test paper URL (Attention Is All You Need)
PAPER_URL = "https://arxiv.org/abs/1706.03762"

@pytest.fixture
def real_paper_data():
    """Create real paper data from the actual paper."""
    paper = ArXivPaper(PAPER_URL)
    return paper.get_paper_data()

@pytest.fixture
def paper_summarizer():
    """Create a PaperSummarizer instance with real API key."""
    return PaperSummarizer(openai_api_key=os.getenv("OPENAI_API_KEY"))

def test_paper_summarizer_initialization(paper_summarizer):
    """Test that PaperSummarizer initializes correctly."""
    assert paper_summarizer._llm is not None
    assert paper_summarizer._output_parser is not None
    assert paper_summarizer._prompt_template is not None

def test_prepare_prompt_inputs(paper_summarizer, real_paper_data):
    """Test that prompt inputs are prepared correctly."""
    inputs = paper_summarizer._prepare_prompt_inputs(real_paper_data)
    
    assert inputs["title"] == "Attention Is All You Need"
    assert "Ashish Vaswani" in inputs["authors"]  # First author
    assert "abstract" in inputs
    assert "content" in inputs
    assert "format_instructions" in inputs

def test_create_prompt_template(paper_summarizer):
    """Test that the prompt template is created correctly."""
    template = paper_summarizer._create_prompt_template()
    
    assert template is not None
    assert len(template.messages) == 2  # System and Human messages

def test_generate_summary(paper_summarizer, real_paper_data):
    """Test that generate_summary produces a valid markdown output."""
    # Generate the summary
    markdown = paper_summarizer.generate_summary(real_paper_data)
    
    # Verify the markdown structure
    assert "# Summary of Attention Is All You Need" in markdown
    assert "## Authors" in markdown
    assert "Ashish Vaswani" in markdown  # First author
    assert "## Abstract" in markdown
    assert "## Key Points" in markdown
    assert "## Methodology" in markdown
    assert "## Results" in markdown
    assert "## Implications" in markdown

def test_generate_summary_with_empty_paper_data(paper_summarizer):
    """Test that generate_summary handles empty paper data appropriately."""
    empty_paper_data = {
        "details": {
            "title": "",
            "authors": [],
            "abstract": "",
            "arxiv_id": "",
            "published": "",
            "categories": [],
            "doi": "",
            "pdf_url": ""
        },
        "documents": []
    }
    
    with pytest.raises(Exception):
        paper_summarizer.generate_summary(empty_paper_data)

def test_generate_summary_with_missing_fields(paper_summarizer):
    """Test that generate_summary handles missing fields appropriately."""
    incomplete_paper_data = {
        "details": {
            "title": "Test Paper",
            "authors": ["Author 1"]
        },
        "documents": [
            Document(
                page_content="Test content",
                metadata={"chunk_index": 0, "total_chunks": 1}
            )
        ]
    }
    
    with pytest.raises(Exception):
        paper_summarizer.generate_summary(incomplete_paper_data)
