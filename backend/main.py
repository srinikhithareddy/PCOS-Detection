from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS
from routes.auth import router as auth_router
from routes.quality import router as quality_router
from routes.preprocessing import router as preprocessing_router
from routes.clinical_preprocessing import router as clinical_preprocessing_router
from routes.pipeline import router as pipeline_router

app = FastAPI(title="X-OVAVISION-ULTRA API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(quality_router, prefix="/api")
app.include_router(preprocessing_router, prefix="/api")
app.include_router(clinical_preprocessing_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "ok", "message": "X-OVAVISION-ULTRA API is running"}
