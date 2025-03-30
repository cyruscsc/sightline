from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    paper_url: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "paper_url": "https://arxiv.org/abs/1706.03762",
                }
            ]
        }
    }


class SummaryResponse(BaseModel):
    title: str = Field(description="The title of the paper")
    authors: list[str] = Field(description="List of paper authors")
    abstract: str = Field(description="A concise summary of the paper's abstract")
    key_points: list[str] = Field(description="Key points and findings from the paper")
    methodology: str = Field(description="Description of the methodology used")
    results: str = Field(description="Main results and conclusions")
    implications: str = Field(
        description="Implications and potential impact of the research"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Attention Is All You Need",
                    "authors": [
                        "Ashish Vaswani",
                        "Noam Shazeer",
                        "Niki Parmar",
                        "Jakob Uszkoreit",
                        "Llion Jones",
                        "Aidan N. Gomez",
                        "Łukasz Kaiser",
                        "Illia Polosukhin",
                    ],
                    "abstract": "The Transformer architecture introduces a novel approach to sequence transduction tasks, relying entirely on attention mechanisms without recurrence or convolutions.",
                    "key_points": [
                        "Introduces the Transformer architecture",
                        "Achieves state-of-the-art results on machine translation",
                        "More parallelizable and requires less training time than RNNs",
                    ],
                    "methodology": "The model uses stacked encoder-decoder layers with multi-head self-attention and position-wise fully connected feed-forward networks.",
                    "results": "Achieves 28.4 BLEU on WMT 2014 English-to-German translation, surpassing all previously published results.",
                    "implications": "Presents a new paradigm for sequence modeling that could be applied to various NLP tasks beyond translation.",
                }
            ]
        }
    }
