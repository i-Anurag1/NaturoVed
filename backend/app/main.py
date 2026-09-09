"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import health, predict, history, weather, dashboard, gradcam, report

app = FastAPI(
    title="Multimodal Crop Disease Diagnosis API",
    description="Leaf Image + Field Context -> Disease + Severity + Field Risk + Recommendations",
    version="2.0.0",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve uploaded images statically so the frontend can preview them ---
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def on_startup():
    init_db()
    print("[startup] Database initialized.")
    print(f"[startup] Environment: {settings.APP_ENV}")
    print(f"[startup] CORS origins: {settings.cors_origins_list}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(weather.router)
app.include_router(dashboard.router)
app.include_router(gradcam.router)
app.include_router(report.router)


@app.get("/")
def root():
    return {
        "message": "Multimodal Crop Disease Diagnosis API",
        "docs": "/docs",
        "health": "/health",
    }
