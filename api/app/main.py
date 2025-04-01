from app.router import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(
    title="SightLine API",
    description="Get insights from arXiv papers",
    version="0.1.0",
    root_path="/api/v1",
)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://sightline.hatchside.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
