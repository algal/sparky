"""
Provider middleware that uses OpenClaw Gateway for LLM responses.

Replaces ProviderMiddleware (Anthropic direct) with OpenClaw Gateway streaming.
OpenClaw manages conversation history and tool execution server-side, so this
middleware is simpler: no tool re-entrancy, no local history tracking.

Drop-in replacement: exposes the same set_user_message() interface so the
state machine's _process_request() works unchanged.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional

from sparky_mvp.core.streaming import StreamMiddleware, StreamChunk
from sparky_mvp.core.openclaw_gateway_client import (
    OpenClawGatewayClient,
    OpenClawEventType,
)

logger = logging.getLogger(__name__)


class OpenClawProviderMiddleware(StreamMiddleware):
    """
    Middleware that streams LLM responses from the OpenClaw Gateway.

    Like ProviderMiddleware, this is a *source* middleware: it ignores the input
    stream and yields StreamChunks from the Gateway's chat.send response.

    OpenClaw handles tools server-side, so ToolCallMiddleware is NOT needed
    in the pipeline when using this provider.
    """

    def __init__(
        self,
        gateway_client: OpenClawGatewayClient,
        session_key: str = "agent:main:reachy",
        timeout_ms: int = 60000,
    ):
        self.gateway_client = gateway_client
        self.session_key = session_key
        self.timeout_ms = timeout_ms
        self.user_message: Optional[str] = None
        self.speaker_name: Optional[str] = None

    def set_user_message(self, message: str) -> None:
        """Set the user message for the next chat.send call."""
        self.user_message = message
        logger.debug("OpenClaw provider message set: %s...", message[:50])

    def set_speaker_name(self, name: Optional[str]) -> None:
        """Set the identified speaker for the next chat.send call."""
        self.speaker_name = name

    def _build_voice_message(self) -> str:
        """Build the prefixed message for chat.send.

        Prepends a [speaker:name] tag (or [speaker:unknown]) so the agent
        knows who is speaking.  Voice-mode formatting guidance lives in
        SOUL.md and does not need to be repeated per-turn.
        """
        speaker = self.speaker_name or "unknown"
        return f"[speaker:{speaker}] {self.user_message}"

    async def process(
        self, stream: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """
        Send user message to OpenClaw Gateway and yield streaming chunks.

        Ignores the input stream (source middleware).
        """
        if not self.user_message:
            logger.warning("OpenClaw provider called with no user message")
            yield StreamChunk.error("No user message set")
            return

        logger.info(
            "OpenClaw chat.send: session=%s msg=%s",
            self.session_key,
            self.user_message[:80],
        )

        text_chars = 0

        try:
            async for event in self.gateway_client.chat_send(
                session_key=self.session_key,
                message=self._build_voice_message(),
                timeout_ms=self.timeout_ms,
            ):
                if event.type == OpenClawEventType.CHAT_DELTA:
                    text_chars += len(event.text)
                    yield StreamChunk.text(event.text)

                elif event.type == OpenClawEventType.TOOL_TRACE:
                    logger.info("OpenClaw tool: %s", event.tool_name)
                    yield StreamChunk.tool_notification(
                        event.tool_name, event.tool_args
                    )

                elif event.type == OpenClawEventType.CHAT_FINAL:
                    # Informational — one LLM step done. Don't terminate;
                    # there may be more steps (tool use → another LLM call).
                    logger.info(
                        "OpenClaw chat_final: text_chars=%d", text_chars
                    )

                elif event.type == OpenClawEventType.LIFECYCLE_END:
                    logger.info(
                        "OpenClaw lifecycle_end: text_chars=%d", text_chars
                    )

                elif event.type == OpenClawEventType.ERROR:
                    logger.error("OpenClaw error: %s", event.error_message)
                    yield StreamChunk.error(event.error_message)
                    return

            # Gateway client generator completed (drain finished) — emit finish
            logger.info(
                "OpenClaw stream complete: text_chars=%d", text_chars
            )
            yield StreamChunk.finish("stop")

        except asyncio.CancelledError:
            logger.info("OpenClaw provider cancelled")
            raise

        except Exception as e:
            logger.error("OpenClaw provider error: %s", e, exc_info=True)
            yield StreamChunk.error(f"OpenClaw error: {e}")
