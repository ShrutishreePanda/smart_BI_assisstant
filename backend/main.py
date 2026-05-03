from __future__ import annotations
from fastapi import FastAPI

from backend.routes.upload import router as upload_router
from backend.routes.eda import eda_router
from backend.routes.clean import router as clean_router
from backend.routes.ml import router as ml_router
from backend.routes.visualization import router as vis_router

app = FastAPI(title="Smart BI Assistant API")

app.include_router(upload_router)
app.include_router(eda_router)
app.include_router(clean_router)
app.include_router(ml_router)
app.include_router(vis_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}