import asyncio
import io
import json

import torchaudio as ta
import websockets

from chatterbox_vllm.tts import ChatterboxTTS


async def handle_connection(websocket):
    async for message in websocket:
        data = json.loads(message)
        prompt = data.get("prompt", "")
        audio_prompt = data.get("audio_prompt_path")
        exaggeration = data.get("exaggeration", 0.5)

        stream = model.generate_stream(
            prompt,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
        )

        for chunk in stream:
            buf = io.BytesIO()
            ta.save(buf, chunk, model.sr, format="mp3")
            await websocket.send(buf.getvalue())
        await websocket.send("done")


async def main():
    global model
    model = ChatterboxTTS.from_pretrained()
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
