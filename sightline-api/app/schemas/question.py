from pydantic import BaseModel


class QuestionRequest(BaseModel):
    paper_url: str
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "paper_url": "https://arxiv.org/abs/1706.03762",
                    "question": "What is the proposed network architecture?",
                }
            ]
        }
    }


class QuestionResponse(BaseModel):
    answer: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The proposed network architecture is the Transformer architecture. This model primarily uses a mechanism known as self-attention, combined with feed-forward neural networks"
                }
            ]
        }
    }
