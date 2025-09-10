import asyncio
import json

import websockets


async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"prompt": "streaming test"}))
        idx = 0
        while True:
            message = await ws.recv()
            if isinstance(message, str) and message == "done":
                break
            with open(f"ws-chunk-{idx}.mp3", "wb") as f:
                f.write(message)
            idx += 1
        print(f"received {idx} chunks")


if __name__ == "__main__":
    asyncio.run(main())
