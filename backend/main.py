from __future__ import annotations
from fastapi import FastAPI

from backend.routes.upload import router as upload_router
from backend.routes.eda import eda_router

app = FastAPI(title="Smart BI Assistant API")

app.include_router(upload_router)
app.include_router(eda_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
