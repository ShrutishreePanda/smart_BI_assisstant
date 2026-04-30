from __future__ import annotations

from fastapi import FastAPI

from routes.eda import router as eda_router
from routes.ml import router as ml_router


app = FastAPI(title="Smart BI Assistant API")

app.include_router(eda_router)
app.include_router(ml_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
