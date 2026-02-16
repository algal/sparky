#!/usr/bin/env python3
"""Dump all raw WS frames from an OpenClaw Gateway chat.send.

Shows exactly what the gateway sends — no filtering, no interpretation.

Usage:
    PYTHONPATH=. .venv/bin/python3 tools/gateway_dump.py "What time is it?"
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import websockets


async def main():
    msg = " ".join(sys.argv[1:]) or "Hello"

    # Load token
    token = ""
    oc_json = Path.home() / ".openclaw" / "openclaw.json"
    if oc_json.exists():
        with open(oc_json) as f:
            data = json.load(f)
        token = data.get("gateway", {}).get("auth", {}).get("token", "")

    url = "ws://127.0.0.1:18789"
    ws = await websockets.connect(url)

    # Handshake
    raw = await asyncio.wait_for(ws.recv(), timeout=10)
    challenge = json.loads(raw)
    assert challenge["event"] == "connect.challenge"

    connect_req = {
        "type": "req", "id": "r1", "method": "connect",
        "params": {
            "minProtocol": 3, "maxProtocol": 3,
            "client": {"id": "gateway-client", "version": "0.1.0", "platform": "linux", "mode": "backend"},
            "role": "operator", "scopes": ["operator.read", "operator.write"],
            "caps": [], "commands": [], "permissions": {},
            "auth": {"token": token} if token else {},
            "locale": "en-US", "userAgent": "gateway-dump/0.1",
        },
    }
    await ws.send(json.dumps(connect_req))
    raw = await asyncio.wait_for(ws.recv(), timeout=10)
    res = json.loads(raw)
    assert res["ok"], f"Connect failed: {res}"

    # Drain initial events
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.3)
        except asyncio.TimeoutError:
            break

    # Send chat
    turn_id = f"turn_{uuid.uuid4()}"
    chat_req = {
        "type": "req", "id": "r2", "method": "chat.send",
        "params": {"sessionKey": "agent:main:reachy", "message": msg, "idempotencyKey": turn_id, "timeoutMs": 60000},
    }
    await ws.send(json.dumps(chat_req))
    print(f">>> {msg}\n")

    # Read ALL frames until lifecycle:end + drain
    lifecycle_ended = False
    frame_num = 0
    while True:
        timeout = 2.0 if lifecycle_ended else 70.0
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            if lifecycle_ended:
                print("\n=== drain timeout ===")
            else:
                print("\n=== timeout ===")
            break

        frame = json.loads(raw)
        frame_num += 1
        ftype = frame.get("type", "")
        event = frame.get("event", "")

        if ftype == "res":
            print(f"[{frame_num:3d}] RES id={frame['id']} ok={frame.get('ok')}")
            continue

        if ftype == "event" and event == "chat":
            payload = frame.get("payload", {})
            state = payload.get("state", "")
            msg_obj = payload.get("message", {})
            content = msg_obj.get("content", [])
            # Show content blocks
            print(f"[{frame_num:3d}] EVENT:CHAT state={state}")
            for i, block in enumerate(content):
                if isinstance(block, dict):
                    btype = block.get("type", "?")
                    if btype == "text":
                        text = block.get("text", "")
                        print(f"       content[{i}] type=text len={len(text)}: {text[:120]!r}")
                    else:
                        print(f"       content[{i}] type={btype} keys={list(block.keys())}")
                else:
                    print(f"       content[{i}] raw={str(block)[:100]!r}")
            continue

        if ftype == "event" and event == "agent":
            payload = frame.get("payload", {})
            stream = payload.get("stream", "")
            data = payload.get("data", {})
            if stream == "lifecycle":
                phase = data.get("phase", "")
                print(f"[{frame_num:3d}] EVENT:AGENT lifecycle:{phase}")
                if phase == "end":
                    lifecycle_ended = True
            elif stream == "assistant":
                # Show assistant text
                text = data.get("text", "")
                print(f"[{frame_num:3d}] EVENT:AGENT assistant len={len(text)}: {text[:120]!r}")
            elif stream == "tool":
                name = data.get("name", "")
                print(f"[{frame_num:3d}] EVENT:AGENT tool:{name} data_keys={list(data.keys())}")
            else:
                print(f"[{frame_num:3d}] EVENT:AGENT stream={stream} data_keys={list(data.keys())}")
            continue

        print(f"[{frame_num:3d}] {ftype}:{event} (ignored)")

    await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
