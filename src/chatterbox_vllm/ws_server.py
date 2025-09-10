import asyncio
import json
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:  # pragma: no cover - avoid heavy import at runtime
    from .tts import ChatterboxTTS


async def _handle_connection(websocket: websockets.WebSocketServerProtocol, tts: 'ChatterboxTTS') -> None:
    async for message in websocket:
        data = json.loads(message)
        text = data.get("text", "")
        # Send sample rate so the client knows how to decode chunks
        await websocket.send(json.dumps({"sr": tts.sr}))
        for _, chunk in tts.generate(text, stream=True):
            await websocket.send(chunk.cpu().numpy().astype("float32").tobytes())
        await websocket.send(json.dumps({"event": "end"}))


async def serve_tts(tts: 'ChatterboxTTS', host: str = "0.0.0.0", port: int = 8765) -> None:
    async def handler(websocket):
        await _handle_connection(websocket, tts)

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever


def main() -> None:
    import argparse
    import os

    # Ensure our custom tokenizer remains visible when vLLM launches
    # its engine. Running without multiprocessing avoids the child
    # process losing the TokenizerRegistry entries.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from .tts import ChatterboxTTS

    parser = argparse.ArgumentParser(description="Chatterbox vLLM TTS WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Local checkpoint directory")
    args = parser.parse_args()

    if args.ckpt_dir:
        tts = ChatterboxTTS.from_local(args.ckpt_dir)
    else:
        tts = ChatterboxTTS.from_pretrained()
    asyncio.run(serve_tts(tts, host=args.host, port=args.port))


if __name__ == "__main__":
    main()
