import pytest
from app.paper_reader.arxiv_paper import ArXivPaper
from app.paper_reader.paper_qa import PaperQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test paper URL (Attention Is All You Need)
PAPER_URL = "https://arxiv.org/abs/1706.03762"


@pytest.fixture
def paper_qa():
    """Fixture to create a PaperQA instance with the test paper."""
    arxiv_paper = ArXivPaper(PAPER_URL)
    paper_data = arxiv_paper.get_paper_data()
    return PaperQA(paper_data)


def test_paper_qa_initialization(paper_qa):
    """Test that PaperQA initializes correctly with paper data."""
    assert paper_qa is not None
    assert paper_qa.documents is not None
    assert len(paper_qa.documents) > 0
    assert paper_qa.paper_details is not None


def test_paper_details(paper_qa):
    """Test that paper details are correctly retrieved."""
    details = paper_qa.paper_details
    # Check that the arxiv_id contains the expected ID (it may include version and full URL)
    assert "1706.03762" in details["arxiv_id"]
    assert "Attention Is All You Need" in details["title"]
    assert len(details["authors"]) > 0
    assert "abstract" in details
    assert "pdf_url" in details


def test_question_answering(paper_qa):
    """Test that the QA system can answer questions about the paper."""
    # Test basic question about the paper's main contribution
    question = "What is the main contribution of this paper?"
    response = paper_qa.ask_question(question)
    assert response is not None
    assert "answer" in response
    answer = response["answer"]
    assert len(answer) > 0
    assert "transformer" in answer.lower()

    # Test specific technical question
    question = "What is self-attention and how does it work?"
    response = paper_qa.ask_question(question)
    assert response is not None
    assert "answer" in response
    answer = response["answer"]
    assert len(answer) > 0
    assert "attention" in answer.lower()

    # Test question about architecture
    question = "What is the architecture of the Transformer model?"
    response = paper_qa.ask_question(question)
    assert response is not None
    assert "answer" in response
    answer = response["answer"]
    assert len(answer) > 0
    assert "encoder" in answer.lower() or "decoder" in answer.lower()


def test_irrelevant_question(paper_qa):
    """Test handling of questions not related to the paper."""
    question = "What is the capital of France?"
    response = paper_qa.ask_question(question)
    assert response is not None
    assert "answer" in response
    answer = response["answer"]
    assert "not relevant" in answer.lower()


def test_empty_question(paper_qa):
    """Test handling of empty questions."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        paper_qa.ask_question("")
