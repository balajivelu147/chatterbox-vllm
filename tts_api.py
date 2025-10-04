#!/usr/bin/env python3
"""FastAPI application that exposes the Chatterbox TTS model over HTTP."""

from __future__ import annotations

import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chatterbox_vllm.tts import ChatterboxTTS


class TTSRequest(BaseModel):
    """Request payload for text-to-speech synthesis."""

    text: str = Field(..., description="Text to convert to speech.")
    audio_prompt_path: Optional[str] = Field(
        default=None,
        description="Optional path to an audio file that should be used as a voice reference.",
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Controls how expressive the generated speech should be.",
    )
    temperature: float = Field(default=0.8, gt=0.0, description="Sampling temperature passed to the language model.")
    top_p: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Top-p nucleus sampling parameter used for text generation.",
    )
    repetition_penalty: float = Field(
        default=2.0,
        gt=0.0,
        description="Penalty applied to repeated tokens during text generation.",
    )
    diffusion_steps: int = Field(
        default=10,
        ge=1,
        description="Number of diffusion steps used when vocoding the final waveform.",
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Upper bound on the number of speech tokens produced by the language model.",
    )


def _synthesize_speech(model: ChatterboxTTS, request: TTSRequest) -> io.BytesIO:
    prompts = [request.text]
    audios = model.generate(
        prompts,
        audio_prompt_path=request.audio_prompt_path,
        exaggeration=request.exaggeration,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        diffusion_steps=request.diffusion_steps,
        max_tokens=request.max_tokens,
    )

    if not audios:
        raise RuntimeError("TTS generation returned no audio segments.")

    waveform = audios[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform.cpu(), model.sr, format="wav")
    buffer.seek(0)
    return buffer


def _resolve_max_workers() -> int:
    workers_env = os.getenv("CHATTERBOX_TTS_WORKERS")
    if workers_env is None:
        return 4
    try:
        value = int(workers_env)
    except ValueError as exc:
        raise ValueError("CHATTERBOX_TTS_WORKERS must be an integer") from exc
    return max(1, value)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=_resolve_max_workers())
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(
                executor,
                lambda: ChatterboxTTS.from_pretrained(
                    max_model_len=1000,
                ),
            )
        except Exception:
            executor.shutdown(wait=True, cancel_futures=True)
            raise

        app.state.executor = executor
        app.state.model = model

        try:
            yield
        finally:
            model = getattr(app.state, "model", None)
            if model is not None:
                model.shutdown()
            executor = getattr(app.state, "executor", None)
            if executor is not None:
                executor.shutdown(wait=True)

    app = FastAPI(title="Chatterbox TTS API", lifespan=lifespan)

    @app.post("/tts", response_class=StreamingResponse)
    async def synthesize(request: TTSRequest) -> StreamingResponse:
        model: Optional[ChatterboxTTS] = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="TTS model is still loading. Please try again later.")

        loop = asyncio.get_running_loop()
        try:
            audio_buffer = await loop.run_in_executor(app.state.executor, _synthesize_speech, model, request)
        except Exception as exc:  # noqa: BLE001 - make sure we surface errors cleanly
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        headers = {"Content-Disposition": "inline; filename=output.wav"}
        return StreamingResponse(audio_buffer, media_type="audio/wav", headers=headers)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("tts_api:app", host="0.0.0.0", port=8021, reload=False)
