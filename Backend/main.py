from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


class dataformat(BaseModel):
    response1 : str
    response2 : str

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "Frontend" / "index.html")


