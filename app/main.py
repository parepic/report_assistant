from fastapi import FastAPI

from app.api.routes import chat, compare, documents, risk_factors

app = FastAPI()

app.include_router(chat.router)
app.include_router(compare.router)
app.include_router(documents.router)
app.include_router(risk_factors.router)
