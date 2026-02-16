"""Sentence buffering middleware."""

import logging
from typing import AsyncIterator, Optional
from sparky_mvp.core.streaming import StreamMiddleware, StreamChunk

logger = logging.getLogger(__name__)


class SentenceBufferMiddleware(StreamMiddleware):
    """
    Buffer text chunks until complete sentences are formed.

    Yields sentence chunks when sentence-ending punctuation is detected (. ! ?).
    This ensures downstream middlewares (like TTS) receive complete sentences
    rather than individual characters or words.

    Non-text chunks (tool_call, tool_notification, finish, error) are passed through immediately.
    """

    def __init__(self):
        """Initialize with sentence-ending punctuation set."""
        self.sentence_endings = {'.', '!', '?'}

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream and yield complete sentences.

        Args:
            stream: Input stream of chunks

        Yields:
            StreamChunk objects - text chunks are buffered into sentences,
            other chunk types are passed through immediately
        """
        accumulated = ""
        last_text_chunk: Optional[StreamChunk] = None

        #         print(f"[BUFFER] SentenceBufferMiddleware.process() starting", flush=True)
        async for chunk in stream:
        #             print(f"[BUFFER] Received chunk: type={chunk.type}, content='{chunk.content[:30] if chunk.content else ''}'", flush=True)
            # Pass through non-text chunks immediately (tool_call, tool_notification, finish, etc.)
            if chunk.type != "text":
                # Flush any accumulated text before passing through
                if accumulated.strip() and last_text_chunk:
                    logger.debug(f"Flushing accumulated text before {chunk.type}: {len(accumulated)} chars")
                    yield StreamChunk.sentence(
                        content=accumulated,
                        role=last_text_chunk.role
                    )
                    accumulated = ""

                # Pass through the non-text chunk
                logger.debug(f"Passing through {chunk.type} chunk")
                yield chunk
                continue

            # Buffer text chunks
            last_text_chunk = chunk
            accumulated += chunk.content
            logger.debug(f"Accumulated text: {len(accumulated)} chars")
        #             print(f"[BUFFER] Accumulated: '{accumulated}'", flush=True)

            # Check if we have sentence ending
            has_sentence_ending = any(
                accumulated.rstrip().endswith(ending)
                for ending in self.sentence_endings
            )

            if has_sentence_ending:
                if accumulated.strip():
                    logger.info(f"Complete sentence detected: {accumulated[:50]}...")
        #                     print(f"[BUFFER] ✓ Complete sentence! Yielding sentence chunk: '{accumulated}'", flush=True)
                    yield StreamChunk.sentence(
                        content=accumulated,
                        role=chunk.role
                    )
                    accumulated = ""

        # Flush remaining content at end of stream
        if accumulated.strip() and last_text_chunk:
            logger.debug(f"Flushing final accumulated text: {len(accumulated)} chars")
        #             print(f"[BUFFER] Flushing final text as sentence: '{accumulated}'", flush=True)
            yield StreamChunk.sentence(
                content=accumulated,
                role=last_text_chunk.role
            )