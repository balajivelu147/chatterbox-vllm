#!/usr/bin/env python3
"""FastAPI application that exposes the Chatterbox TTS model over HTTP."""

from __future__ import annotations

import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch

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
    max_tokens = min(request.max_tokens, model.max_model_len)

    audios = model.generate(
        prompts,
        audio_prompt_path=request.audio_prompt_path,
        exaggeration=request.exaggeration,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        diffusion_steps=request.diffusion_steps,
        max_tokens=max_tokens,
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


def _resolve_base_max_model_len() -> int:
    raw_value = os.getenv("CHATTERBOX_TTS_MAX_MODEL_LEN")
    if raw_value is None:
        return 1000
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError("CHATTERBOX_TTS_MAX_MODEL_LEN must be an integer") from exc
    return max(64, parsed)


def _max_model_len_candidates(base: int) -> List[int]:
    base = max(64, base)
    values = {base}

    step = 128
    cursor = base
    while cursor - step >= 64:
        cursor -= step
        values.add(cursor)

    cursor = base
    while cursor > 64:
        cursor = max(cursor // 2, 64)
        values.add(cursor)
        if cursor == 64:
            break

    return sorted(values, reverse=True)


def _gpu_utilization_candidates() -> List[float]:
    override = os.getenv("CHATTERBOX_VLLM_GPU_UTILIZATION")
    if override is not None:
        try:
            return [float(override)]
        except ValueError as exc:
            raise ValueError(
                "CHATTERBOX_VLLM_GPU_UTILIZATION must be a floating point value"
            ) from exc

    return [
        0.50,
        0.45,
        0.40,
        0.36,
        0.32,
        0.28,
        0.25,
        0.22,
        0.20,
        0.18,
        0.16,
        0.14,
        0.12,
        0.10,
    ]


def _load_model_with_backoff() -> ChatterboxTTS:
    base_max_len = _resolve_base_max_model_len()
    max_len_candidates = _max_model_len_candidates(base_max_len)
    gpu_util_candidates = _gpu_utilization_candidates()

    prior_gpu_env = os.getenv("CHATTERBOX_VLLM_GPU_UTILIZATION")
    last_error: Optional[Exception] = None
    attempt_messages: List[str] = []

    for gpu_util in gpu_util_candidates:
        gpu_util_str = f"{gpu_util:.3f}"
        if prior_gpu_env is None:
            os.environ["CHATTERBOX_VLLM_GPU_UTILIZATION"] = gpu_util_str

        for max_len in max_len_candidates:
            print(
                "Attempting to load Chatterbox TTS "
                f"(max_model_len={max_len}, gpu_util={gpu_util_str})"
            )
            try:
                model = ChatterboxTTS.from_pretrained(
                    max_batch_size=1,
                    max_model_len=max_len,
                )
                if prior_gpu_env is None:
                    os.environ["CHATTERBOX_VLLM_GPU_UTILIZATION"] = gpu_util_str
                return model
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempt_messages.append(
                    f"max_model_len={max_len}, gpu_util={gpu_util_str}: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if prior_gpu_env is None:
            os.environ["CHATTERBOX_VLLM_GPU_UTILIZATION"] = gpu_util_str

    if prior_gpu_env is not None:
        os.environ["CHATTERBOX_VLLM_GPU_UTILIZATION"] = prior_gpu_env
    else:
        os.environ.pop("CHATTERBOX_VLLM_GPU_UTILIZATION", None)

    details = "\n".join(attempt_messages[-3:])
    raise RuntimeError(
        "Failed to initialize the Chatterbox TTS model after trying multiple "
        "GPU memory budgets and context lengths. "
        f"Last error: {last_error}. Recent attempts:\n{details}"
    ) from last_error


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=_resolve_max_workers())
        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(
                executor,
                _load_model_with_backoff,
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
