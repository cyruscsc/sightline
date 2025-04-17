from app.schemas import (
    HealthCheck,
    SummaryRequest,
    SummaryResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.paper_reader import ArXivPaper, PaperSummarizer, PaperQA
from datetime import datetime
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/", response_model=HealthCheck)
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(summary_request: SummaryRequest) -> dict:
    try:
        # Initialize paper reader and get paper data
        paper = ArXivPaper(summary_request.paper_url, 8000, 800)
        paper_data = paper.get_paper_data()

        # Generate summary
        summarizer = PaperSummarizer()
        summary = summarizer.generate_summary(paper_data)

        return summary

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing paper: {str(e)}")


@router.post("/ask", response_model=QuestionResponse)
async def ask(question_request: QuestionRequest) -> dict:
    try:
        # Initialize paper reader and get paper data
        paper = ArXivPaper(question_request.paper_url)
        paper_data = paper.get_paper_data()

        # Initialize QA system and get answer
        qa_system = PaperQA(paper_data)
        answer = qa_system.ask_question(question_request.question)

        return answer

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )
