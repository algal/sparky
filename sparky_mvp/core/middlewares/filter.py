"""Filter middleware for stream processing."""

from typing import AsyncIterator, Callable
from sparky_mvp.core.streaming import StreamMiddleware, StreamChunk


class FilterMiddleware(StreamMiddleware):
    """
    Filter chunks based on a predicate function.

    Useful for removing empty chunks, filtering specific patterns,
    or conditionally passing through content.
    """

    def __init__(self, predicate: Callable[[StreamChunk], bool]):
        """
        Initialize filter with a predicate function.

        Args:
            predicate: Function that returns True if chunk should be yielded,
                      False to filter it out. Receives a StreamChunk as input.

        Example:
            # Filter out empty chunks
            FilterMiddleware(lambda c: bool(c.content.strip()))
        """
        self.predicate = predicate

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream and yield only chunks that pass the predicate.

        Args:
            stream: Input stream of chunks

        Yields:
            StreamChunk objects that satisfy the predicate
        """
        async for chunk in stream:
            if self.predicate(chunk):
                yield chunk
            else:
                pass  # Filtered out
