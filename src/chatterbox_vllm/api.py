"""FastAPI application exposing the Chatterbox TTS model over HTTP."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .tts import ChatterboxTTS, REPO_ID


class SynthesisRequest(BaseModel):
    """Payload accepted by the ``/tts`` endpoint."""

    text: str = Field(..., min_length=1, description="Text to be synthesized")
    audio_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional path to a reference audio clip used for voice cloning.",
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotion exaggeration factor passed to the model.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        description="Sampling temperature used during token generation.",
    )
    diffusion_steps: int = Field(
        default=10,
        ge=1,
        description="Number of diffusion steps for the vocoder stage.",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of speech tokens to generate.",
    )


class HealthResponse(BaseModel):
    status: str
    sample_rate: int


_logger = logging.getLogger(__name__)


_tts_instance: ChatterboxTTS | None = None
_tts_lock = asyncio.Lock()


def _create_executor() -> ThreadPoolExecutor:
    max_workers = int(os.getenv("CHATTERBOX_API_MAX_WORKERS", "4"))
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts-worker")


_executor = _create_executor()


def _candidate_devices() -> list[str]:
    requested = os.getenv("CHATTERBOX_TARGET_DEVICE")
    if requested:
        return [requested]

    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def _load_tts() -> ChatterboxTTS:
    ckpt_dir = os.getenv("CHATTERBOX_CKPT_DIR")
    repo_id = os.getenv("CHATTERBOX_REPO_ID", REPO_ID)
    revision = os.getenv("CHATTERBOX_REVISION", "1b475dffa71fb191cb6d5901215eb6f55635a9b6")

    attempts = []
    for device in _candidate_devices():
        try:
            if ckpt_dir:
                return ChatterboxTTS.from_local(
                    ckpt_dir,
                    target_device=device,
                )

            return ChatterboxTTS.from_pretrained(
                repo_id=repo_id,
                revision=revision,
                target_device=device,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.exception("Failed to initialize ChatterboxTTS on %s", device)
            attempts.append(f"{device}: {exc}")

    details = "; ".join(attempts) if attempts else "no devices attempted"
    raise RuntimeError(f"Unable to initialize ChatterboxTTS ({details})")


async def get_tts() -> ChatterboxTTS:
    """Lazily instantiate the TTS model, ensuring a single shared instance."""

    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    async with _tts_lock:
        if _tts_instance is None:
            loop = asyncio.get_running_loop()
            try:
                _tts_instance = await loop.run_in_executor(_executor, _load_tts)
            except Exception as exc:  # pragma: no cover - defensive
                _logger.exception("Unable to load ChatterboxTTS")
                raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _tts_instance


def _synthesize(tts: ChatterboxTTS, request: SynthesisRequest) -> bytes:
    """Invoke the model and serialize the waveform to a WAV byte stream."""

    wavs = tts.generate(
        prompts=request.text,
        audio_prompt_path=request.audio_prompt_path,
        exaggeration=request.exaggeration,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        diffusion_steps=request.diffusion_steps,
    )

    if not wavs:
        raise RuntimeError("Model returned no audio")

    wav = wavs[0]
    if hasattr(wav, "detach"):
        wav = wav.detach()
    audio = wav.cpu().numpy().astype(np.float32)

    # Normalize to prevent clipping if necessary.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    import soundfile as sf  # type: ignore

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=tts.sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


app = FastAPI(title="Chatterbox TTS API", version="1.0.0")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    _executor.shutdown(wait=True)
    if _tts_instance is not None:
        _tts_instance.shutdown()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    tts = await get_tts()
    return HealthResponse(status="ok", sample_rate=tts.sr)


@app.post("/tts")
async def synthesize(request: SynthesisRequest) -> StreamingResponse:
    tts = await get_tts()
    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(
            _executor,
            partial(_synthesize, tts, request),
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {"Cache-Control": "no-store"}
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers=headers)


@app.exception_handler(ValueError)
async def value_error_handler(_: ValueError):
    return JSONResponse(status_code=400, content={"detail": "Invalid request"})
