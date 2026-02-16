#!/usr/bin/env python3
"""
Smoke test: OpenClawProviderMiddleware through a mini pipeline.

Exercises the full path: OpenClawProviderMiddleware → SentenceBuffer → Filter
and verifies StreamChunk types are correct.

Usage:
    cd forks/reachy-glados-example
    PYTHONPATH=. .venv/bin/python tools/provider_smoke_test.py

Exit codes:
    0 = success
    1 = failure
"""

import asyncio
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("provider_smoke")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sparky_mvp.core.openclaw_gateway_client import create_gateway_client_from_config
from sparky_mvp.core.middlewares.openclaw_provider import OpenClawProviderMiddleware
from sparky_mvp.core.middlewares.sentence_buffer import SentenceBufferMiddleware
from sparky_mvp.core.middlewares.filter import FilterMiddleware
from sparky_mvp.core.streaming import StreamPipeline


async def provider_smoke_test():
    session_key = os.environ.get("OPENCLAW_SESSION_KEY", "reachy")
    message = "Say hello in exactly one short sentence."

    # Create gateway client
    config = {
        "openclaw": {
            "gateway_ws_url": os.environ.get(
                "OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789"
            ),
        }
    }
    client = create_gateway_client_from_config(config)

    log.info("Gateway URL: %s", client.gateway_url)
    log.info("Session key: %s", session_key)

    try:
        await client.connect()
        log.info("Connected (protocol=%s)", client._protocol_version)

        # Build mini pipeline: Provider → SentenceBuffer → Filter
        provider = OpenClawProviderMiddleware(
            gateway_client=client,
            session_key=session_key,
            timeout_ms=30000,
        )
        provider.set_user_message(message)

        pipeline = StreamPipeline([
            provider,
            SentenceBufferMiddleware(),
            FilterMiddleware(
                lambda c: c.type not in ["text", "sentence"] or bool(c.content.strip())
            ),
        ])

        async def empty_stream():
            if False:
                yield

        # Process through pipeline
        chunks_by_type = {}
        accumulated_text = []
        start_time = time.monotonic()
        first_chunk_at = None

        async for chunk in pipeline.process(empty_stream()):
            chunks_by_type[chunk.type] = chunks_by_type.get(chunk.type, 0) + 1

            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
                log.info("First chunk after %.2fs (type=%s)", first_chunk_at - start_time, chunk.type)

            if chunk.type == "text":
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
                accumulated_text.append(chunk.content)
            elif chunk.type == "sentence":
                sys.stdout.write(chunk.content)
                sys.stdout.flush()
                accumulated_text.append(chunk.content)
            elif chunk.type == "tool_notification":
                log.info("Tool notification: %s", chunk.tool_name)
            elif chunk.type == "finish":
                log.info("Finish: reason=%s", chunk.finish_reason)
            elif chunk.type == "error":
                log.error("Error chunk: %s", chunk.content)

        sys.stdout.write("\n")
        sys.stdout.flush()

        elapsed = time.monotonic() - start_time
        full_text = "".join(accumulated_text)

        log.info("--- Provider smoke test ---")
        log.info("Total time: %.2fs", elapsed)
        if first_chunk_at:
            log.info("Time to first chunk: %.2fs", first_chunk_at - start_time)
        log.info("Chunk counts: %s", chunks_by_type)
        log.info("Response: %s", full_text[:200])

        # Validation
        ok = True
        if not full_text:
            log.error("FAIL: No text received")
            ok = False
        if "finish" not in chunks_by_type:
            log.error("FAIL: No finish chunk received")
            ok = False
        if "sentence" not in chunks_by_type and "text" not in chunks_by_type:
            log.error("FAIL: No text or sentence chunks received")
            ok = False

        if ok:
            log.info("SUCCESS: Provider middleware pipeline works end-to-end")
        return ok

    finally:
        await client.close()


if __name__ == "__main__":
    ok = asyncio.run(provider_smoke_test())
    sys.exit(0 if ok else 1)
