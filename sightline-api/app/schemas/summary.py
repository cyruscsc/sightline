from pydantic import BaseModel


class SummaryRequest(BaseModel):
    paper_url: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "paper_url": "https://arxiv.org/pdf/1706.03762",
                }
            ]
        }
    }


class SummaryResponse(BaseModel):
    summary: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": 'The paper "Attention is All You Need" presents the Transformer model, a novel architecture that forgoes traditional sequence-aligned RNNs and instead relies entirely on a mechanism called self-attention.'
                }
            ]
        }
    }
