"""FastAPI main application."""

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, print_config, validate_config
from ..logging_config import logging_context, setup_logging
from .routes import map as map_routes
from .routes import poi, trip

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Trip planner API powered by LangChain/LangGraph",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow common local dev origins on any port (Vite may fallback to 5174/5175).
LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach request id and timing logs."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    origin = request.headers.get("origin", "-")

    with logging_context(request_id=request_id):
        logger.info(
            "http_request_start method=%s path=%s client=%s origin=%s",
            request.method,
            request.url.path,
            client,
            origin,
        )
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "http_request_error method=%s path=%s elapsed_ms=%d",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_end method=%s path=%s status=%s elapsed_ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("\n" + "=" * 60)
    print(f"-> {settings.app_name} v{settings.app_version}")
    print("=" * 60)

    print_config()

    try:
        validate_config()
        print("\n[OK] Configuration validated")
    except ValueError as exc:
        print(f"\n[ERROR] Configuration validation failed:\n{exc}")
        print("\nPlease check .env and ensure required keys are set")
        raise

    print("\n" + "=" * 60)
    print("[DOC] API docs: http://localhost:8000/docs")
    print("[DOC] ReDoc: http://localhost:8000/redoc")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    print("\n" + "=" * 60)
    print("[BYE] Application shutting down...")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
