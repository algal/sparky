#!/usr/bin/env python3
"""
Smoke test: connect to OpenClaw Gateway WS via the client library, stream a response.

Usage:
    cd forks/reachy-glados-example
    PYTHONPATH=. .venv/bin/python tools/gateway_smoke_test.py

Environment:
    OPENCLAW_GATEWAY_TOKEN  -- optional (auto-loaded from ~/.openclaw/openclaw.json)
    OPENCLAW_GATEWAY_URL    -- optional (default: ws://127.0.0.1:18789)
    OPENCLAW_SESSION_KEY    -- optional (default: reachy)

Exit codes:
    0 = success (connected, sent message, received streamed response)
    1 = failure (connection, auth, or protocol error)
"""

import asyncio
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gw_smoke")

# Allow running from repo root with PYTHONPATH=.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sparky_mvp.core.openclaw_gateway_client import (
    OpenClawGatewayClient,
    OpenClawEventType,
    create_gateway_client_from_config,
)


async def smoke_test():
    session_key = os.environ.get("OPENCLAW_SESSION_KEY", "reachy")
    message = "Say hello in exactly one short sentence."

    # Create client (auto-loads token from env or ~/.openclaw/openclaw.json)
    config = {
        "openclaw": {
            "gateway_ws_url": os.environ.get("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789"),
        }
    }
    client = create_gateway_client_from_config(config)

    log.info("Gateway URL: %s", client.gateway_url)
    log.info("Session key: %s", session_key)
    log.info("Message: %s", message)

    try:
        # Connect
        await client.connect()
        log.info("Connected (protocol=%s)", client._protocol_version)

        # Send and stream
        accumulated_text = ""
        first_delta_at = None
        start_time = time.monotonic()

        async for event in client.chat_send(session_key, message, timeout_ms=30000):
            if event.type == OpenClawEventType.CHAT_DELTA:
                if first_delta_at is None:
                    first_delta_at = time.monotonic()
                    log.info("First delta after %.2fs", first_delta_at - start_time)
                accumulated_text += event.text
                sys.stdout.write(event.text)
                sys.stdout.flush()

            elif event.type == OpenClawEventType.CHAT_FINAL:
                if event.text and not accumulated_text:
                    accumulated_text = event.text
                    sys.stdout.write(event.text)
                sys.stdout.write("\n")
                sys.stdout.flush()
                log.info("Got chat final")

            elif event.type == OpenClawEventType.LIFECYCLE_START:
                log.info("Agent lifecycle: start")

            elif event.type == OpenClawEventType.LIFECYCLE_END:
                log.info("Agent lifecycle: end")

            elif event.type == OpenClawEventType.TOOL_TRACE:
                log.info("Agent tool: %s", event.tool_name)

            elif event.type == OpenClawEventType.ERROR:
                log.error("Error: %s", event.error_message)

        elapsed = time.monotonic() - start_time
        log.info("--- Smoke test complete ---")
        log.info("Total time: %.2fs", elapsed)
        if first_delta_at:
            log.info("Time to first byte: %.2fs", first_delta_at - start_time)
        log.info("Response: %s", accumulated_text[:200])

        if accumulated_text:
            log.info("SUCCESS: Got streamed response from OpenClaw Gateway")
            return True
        else:
            log.error("FAIL: No text received")
            return False

    finally:
        await client.close()


if __name__ == "__main__":
    ok = asyncio.run(smoke_test())
    sys.exit(0 if ok else 1)
