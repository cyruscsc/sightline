from pydantic import BaseModel


class HealthCheck(BaseModel):
    status: str
    timestamp: str

    model_config = {
        "json_encoders": {
            "examples": [
                {
                    "status": "ok",
                    "timestamp": "2023-03-29T00:00:00Z",
                }
            ]
        }
    }
