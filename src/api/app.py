"""Local FastAPI demo for the face-crop Xception study."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.decisions import decision_payload, threshold_payload
from src.api.errors import DetectionError
from src.api.scoring import media_kind, resolve_checkpoint, score_media, size_limit

app = FastAPI(title="VeriShield", version="0.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ScoreBody(BaseModel):
    score: float = Field(..., description="Mean fake-class softmax in [0, 1]. Not a file upload.")


def _error(exc: DetectionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/thresholds")
def thresholds():
    """Frozen validation cuts. Does not load a checkpoint or score a file."""
    try:
        return threshold_payload()
    except DetectionError as exc:
        return _error(exc)


@app.post("/api/decision")
def decision(body: ScoreBody):
    """Apply tau1 and tau2 to a supplied score. Does not load a checkpoint."""
    try:
        return decision_payload(body.score)
    except DetectionError as exc:
        return _error(exc)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """Score an image or short video when CHECKPOINT_PATH is a real file."""
    temp_path: Path | None = None
    try:
        kind = media_kind(file.filename or "", file.content_type)
        checkpoint = resolve_checkpoint()
        payload = await file.read()
        if not payload:
            raise DetectionError("The upload was empty. No score was produced.", 400)
        limit = size_limit(kind)
        if len(payload) > limit:
            raise DetectionError(
                "That file is too large for this demo. No score was produced.",
                413,
            )
        suffix = Path(file.filename or "").suffix.lower()
        if not suffix:
            suffix = ".jpg" if kind == "image" else ".mp4"
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(handle.name)
        handle.write(payload)
        handle.close()
        scored = score_media(temp_path, kind, checkpoint)
        body = decision_payload(scored["score"])
        body["n_frames"] = scored["n_frames"]
        body["media_kind"] = scored["media_kind"]
        return body
    except DetectionError as exc:
        return _error(exc)
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=503,
            content={"detail": f"{exc} No score was produced."},
        )
    except (RuntimeError, OSError) as exc:
        message = str(exc)
        if "not scoring" not in message.lower() and "no score" not in message.lower():
            message = f"{message} No score was produced."
        return JSONResponse(status_code=422, content={"detail": message})
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
