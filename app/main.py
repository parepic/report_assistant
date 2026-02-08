from fastapi import FastAPI

from app.api.routes import chat, compare, documents

app = FastAPI()

app.include_router(chat.router)
app.include_router(compare.router)
app.include_router(documents.router)
