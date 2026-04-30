from fastapi import FastAPI
from backend.routes.eda import router as eda_router

app = FastAPI()
app.include_router(eda_router)