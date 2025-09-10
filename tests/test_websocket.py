import asyncio
import json
import numpy as np
import websockets

from chatterbox_vllm.ws_server import serve_tts


class FakeTensor:
    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class DummyTTS:
    sr = 16000

    def generate(self, prompt: str, stream: bool = False):
        assert stream
        for _ in range(2):
            yield 0, FakeTensor(np.zeros(1600, dtype="float32"))


async def run_client():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"text": "hello"}))
        msg = await ws.recv()
        print(msg)
        chunks = 0
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                chunks += 1
            else:
                print(msg)
                break
        print(f"received {chunks} chunks")


async def main():
    tts = DummyTTS()
    server_task = asyncio.create_task(serve_tts(tts))
    await asyncio.sleep(0.1)
    await run_client()
    server_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
