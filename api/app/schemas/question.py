from enum import Enum
from pydantic import BaseModel, Field


class StrategyEnum(str, Enum):
    simple = "simple"
    multi_query = "multi_query"


class QuestionRequest(BaseModel):
    paper_url: str
    question: str
    strategy: StrategyEnum = StrategyEnum.simple

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "paper_url": "https://arxiv.org/abs/1706.03762",
                    "question": "What is the proposed network architecture?",
                    "strategy": "simple",
                }
            ]
        }
    }


class QuestionResponse(BaseModel):
    answer: str = Field(
        description="The answer to the question based on the paper content"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The proposed network architecture is the Transformer architecture. This model primarily uses a mechanism known as self-attention, combined with feed-forward neural networks. The architecture consists of stacked encoder-decoder layers, where each layer contains multi-head self-attention and position-wise fully connected feed-forward networks.",
                }
            ]
        }
    }
