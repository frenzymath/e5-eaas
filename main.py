import os
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI

from embedding import MistralEmbedding


@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    app.state.embedding = MistralEmbedding(os.environ["EMBEDDING_DEVICE"])
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
def embed(q: list[str]) -> list[list[float]]:
    return app.state.embedding.embed(q)
