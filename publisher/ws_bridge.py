import asyncio
import sys

NATS_URL = "nats://127.0.0.1:4222"
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "one"
WS_HOST = "0.0.0.0"
WS_PORT = 4195
WS_PATH = "/get/ws"

async def main():
    import nats
    import websockets
    from websockets.http11 import Request, Response

    clients: set = set()
    lock = asyncio.Lock()

    async def handler(ws):
        async with lock:
            clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            async with lock:
                clients.discard(ws)

    async def on_msg(msg):
        data = msg.data
        async with lock:
            snapshot = list(clients)
        for ws in snapshot:
            try:
                await ws.send(data)
            except Exception:
                pass

    # Only accept connections on /get/ws
    async def process_request(conn, request: Request):
        if request.path != WS_PATH:
            return Response(404, "Not Found", [], b"")
        return None  # continue to handler

    nc = await nats.connect(NATS_URL)
    print(f"Connected to NATS at {NATS_URL}, subscribed to '{SUBJECT}'")
    sub = await nc.subscribe(SUBJECT, cb=on_msg)
    print(f"WebSocket server listening on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")

    async with websockets.serve(
        handler, WS_HOST, WS_PORT,
        process_request=process_request,
    ):
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
