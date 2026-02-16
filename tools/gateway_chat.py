#!/usr/bin/env python3
"""CLI chat client for OpenClaw Gateway — no robot, no STT, no TTS.

Connects to the gateway, sends text, prints streaming events.
For fast iteration on gateway protocol handling.

Usage:
    PYTHONPATH=. .venv/bin/python3 tools/gateway_chat.py
    PYTHONPATH=. .venv/bin/python3 tools/gateway_chat.py --session reachy
    PYTHONPATH=. .venv/bin/python3 tools/gateway_chat.py --once "What time is it?"
"""

import argparse
import asyncio
import logging
import sys
import time

from sparky_mvp.core.openclaw_gateway_client import (
    OpenClawGatewayClient,
    OpenClawEventType,
    create_gateway_client_from_config,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gateway_chat")


async def send_and_print(client: OpenClawGatewayClient, session: str, message: str) -> None:
    """Send a message and print all streaming events."""
    print(f"\n>>> {message}")
    t0 = time.monotonic()
    text_parts = []
    first_text_at = None

    async for event in client.chat_send(session_key=session, message=message):
        elapsed = time.monotonic() - t0

        if event.type == OpenClawEventType.CHAT_DELTA:
            if first_text_at is None:
                first_text_at = elapsed
            text_parts.append(event.text)
            # Print incremental text inline
            sys.stdout.write(event.text)
            sys.stdout.flush()

        elif event.type == OpenClawEventType.CHAT_FINAL:
            logger.info("[%.2fs] chat_final", elapsed)

        elif event.type == OpenClawEventType.LIFECYCLE_START:
            logger.info("[%.2fs] lifecycle:start", elapsed)

        elif event.type == OpenClawEventType.LIFECYCLE_END:
            logger.info("[%.2fs] lifecycle:end", elapsed)

        elif event.type == OpenClawEventType.TOOL_TRACE:
            logger.info("[%.2fs] tool: %s(%s)", elapsed, event.tool_name, event.tool_args)

        elif event.type == OpenClawEventType.ERROR:
            logger.error("[%.2fs] ERROR: %s", elapsed, event.error_message)

    total = time.monotonic() - t0
    full_text = "".join(text_parts)
    print()  # newline after streamed text
    print(f"--- {len(full_text)} chars, TTFB={first_text_at:.2f}s, total={total:.2f}s ---")


async def main():
    parser = argparse.ArgumentParser(description="CLI chat with OpenClaw Gateway")
    parser.add_argument("--session", default="agent:main:reachy", help="Session key (default: agent:main:reachy)")
    parser.add_argument("--once", type=str, help="Send one message and exit")
    parser.add_argument("--url", default="ws://127.0.0.1:18789", help="Gateway WS URL")
    args = parser.parse_args()

    # Load token from ~/.openclaw/openclaw.json
    import json
    from pathlib import Path
    token = ""
    oc_json = Path.home() / ".openclaw" / "openclaw.json"
    if oc_json.exists():
        with open(oc_json) as f:
            data = json.load(f)
        token = data.get("gateway", {}).get("auth", {}).get("token", "")

    client = OpenClawGatewayClient(gateway_url=args.url, token=token)
    await client.connect()

    try:
        if args.once:
            await send_and_print(client, args.session, args.once)
        else:
            print("Connected. Type messages (Ctrl+D to quit).")
            while True:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\nyou> ")
                    )
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.lower() in ("quit", "exit"):
                    break
                await send_and_print(client, args.session, line)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
