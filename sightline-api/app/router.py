from app.schemas.health import HealthCheck
from app.schemas.summary import SummaryRequest, SummaryResponse
from app.schemas.question import QuestionRequest, QuestionResponse
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1",
    tags=["api/v1"],
)


@router.get("/", response_model=HealthCheck)
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(summary_request: SummaryRequest):
    return {"summary": summary_request.paper_url}


@router.post("/ask", response_model=QuestionResponse)
async def ask(question_request: QuestionRequest):
    return {"answer": question_request.question}
