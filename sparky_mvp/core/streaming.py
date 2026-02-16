"""
Async streaming middleware framework for OpenAI-compatible APIs.

Provides a composable middleware pattern for processing streaming responses.
Base classes for building streaming pipelines with custom middleware components.

Middleware implementations are in: sparky_mvp.core.middlewares
"""

from typing import AsyncIterator, List, Dict, Any, Optional
from abc import ABC, abstractmethod


class StreamChunk:
    """
    Represents a single chunk from the stream.

    Chunk types:
    - "text": Raw text content from the model (character/word level)
    - "sentence": Complete sentence after buffering (from SentenceBufferMiddleware)
    - "tool_call": Model is requesting a tool be called
    - "tool_notification": Notification that a tool is being executed (for animations)
    - "tool_result": Result from tool execution (internal use)
    - "tts": Audio data from TTS synthesis (for playback)
    - "finish": Stream completion marker
    - "error": Error occurred during processing
    """

    # Valid chunk types
    VALID_TYPES = {"text", "sentence", "tool_call", "tool_notification", "tool_result", "tts", "finish", "error"}

    def __init__(
        self,
        content: str = "",
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Tool calling support
        type: str = "text",
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
        # TTS support
        audio_data: Optional[bytes] = None
    ):
        """
        Initialize a stream chunk.

        Args:
            content: Text content of the chunk
            role: Role of the message sender (e.g., "assistant", "tool")
            finish_reason: Reason for stream completion (e.g., "stop", "tool_calls")
            metadata: Additional metadata dictionary
            type: Type of chunk (must be in VALID_TYPES)
            tool_name: Name of tool (for tool_call/tool_notification types)
            arguments: Tool arguments (for tool_call types)
            tool_call_id: Unique identifier for tool call
            audio_data: Audio bytes (for tts type chunks)
        """
        if type not in self.VALID_TYPES:
            raise ValueError(f"Invalid chunk type: {type}. Must be one of {self.VALID_TYPES}")

        self._content = content
        self._role = role
        self._finish_reason = finish_reason
        self._metadata = metadata or {}
        self._type = type
        self._tool_name = tool_name
        self._arguments = arguments
        self._tool_call_id = tool_call_id
        self._audio_data = audio_data

    # Properties for read access (maintains backward compatibility)
    @property
    def content(self) -> str:
        """Get chunk content."""
        return self._content

    @property
    def role(self) -> Optional[str]:
        """Get message role."""
        return self._role

    @property
    def finish_reason(self) -> Optional[str]:
        """Get finish reason."""
        return self._finish_reason

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return self._metadata

    @property
    def type(self) -> str:
        """Get chunk type."""
        return self._type

    @property
    def tool_name(self) -> Optional[str]:
        """Get tool name."""
        return self._tool_name

    @property
    def arguments(self) -> Optional[Dict[str, Any]]:
        """Get tool arguments."""
        return self._arguments

    @property
    def tool_call_id(self) -> Optional[str]:
        """Get tool call ID."""
        return self._tool_call_id

    @property
    def audio_data(self) -> Optional[bytes]:
        """Get audio data."""
        return self._audio_data

    # Factory methods for common chunk types
    @classmethod
    def text(cls, content: str, role: Optional[str] = "assistant") -> "StreamChunk":
        """Create a text chunk."""
        return cls(content=content, role=role, type="text")

    @classmethod
    def sentence(cls, content: str, role: Optional[str] = "assistant") -> "StreamChunk":
        """Create a sentence chunk (complete sentence)."""
        return cls(content=content, role=role, type="sentence")

    @classmethod
    def tool_call(cls, tool_name: str, arguments: Dict[str, Any], tool_call_id: str) -> "StreamChunk":
        """Create a tool call chunk."""
        return cls(
            type="tool_call",
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id
        )

    @classmethod
    def tool_notification(cls, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> "StreamChunk":
        """Create a tool notification chunk (for animations)."""
        return cls(
            type="tool_notification",
            tool_name=tool_name,
            arguments=arguments or {}
        )

    @classmethod
    def tool_result(cls, content: str, tool_call_id: str, tool_name: Optional[str] = None) -> "StreamChunk":
        """Create a tool result chunk."""
        return cls(
            type="tool_result",
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            role="tool"
        )

    @classmethod
    def finish(cls, finish_reason: str = "stop") -> "StreamChunk":
        """Create a finish chunk."""
        return cls(type="finish", finish_reason=finish_reason)

    @classmethod
    def error(cls, error_message: str, error_type: Optional[str] = None) -> "StreamChunk":
        """Create an error chunk."""
        metadata = {"error_type": error_type} if error_type else {}
        return cls(type="error", content=error_message, metadata=metadata)

    @classmethod
    def tts(cls, audio_data: bytes, original_text: str = "", metadata: Optional[Dict[str, Any]] = None) -> "StreamChunk":
        """Create a TTS audio chunk."""
        return cls(
            type="tts",
            audio_data=audio_data,
            content=original_text,  # Store original text for reference
            metadata=metadata or {}
        )

    def is_text(self) -> bool:
        """Check if this is a text chunk."""
        return self._type == "text"

    def is_tool_call(self) -> bool:
        """Check if this is a tool call chunk."""
        return self._type == "tool_call"

    def is_tool_notification(self) -> bool:
        """Check if this is a tool notification chunk."""
        return self._type == "tool_notification"

    def is_finish(self) -> bool:
        """Check if this is a finish chunk."""
        return self._type == "finish"

    def is_error(self) -> bool:
        """Check if this is an error chunk."""
        return self._type == "error"

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._type == "tool_call":
            return f"StreamChunk.tool_call(name={self._tool_name!r}, args={self._arguments}, id={self._tool_call_id})"
        elif self._type == "tool_notification":
            return f"StreamChunk.tool_notification(name={self._tool_name!r})"
        elif self._type == "error":
            return f"StreamChunk.error({self._content!r})"
        elif self._type == "finish":
            return f"StreamChunk.finish(reason={self._finish_reason!r})"
        return f"StreamChunk.text({self._content!r}, role={self._role})"


class StreamMiddleware(ABC):
    """Base class for stream middleware."""

    @abstractmethod
    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process the stream and yield transformed chunks.

        Args:
            stream: Input stream of chunks

        Yields:
            Transformed/filtered chunks
        """
        pass


class StreamPipeline:
    """Pipeline for processing streams through multiple middlewares."""

    def __init__(self, middlewares: Optional[List[StreamMiddleware]] = None):
        self.middlewares = middlewares or []

    def add_middleware(self, middleware: StreamMiddleware) -> "StreamPipeline":
        """Add a middleware to the pipeline."""
        self.middlewares.append(middleware)
        return self

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream through all middlewares in order.

        Args:
            stream: Input stream

        Yields:
            Processed chunks
        """
        current_stream = stream

        # Chain middlewares
        for middleware in self.middlewares:
            current_stream = middleware.process(current_stream)

        # Yield from final stream
        async for chunk in current_stream:
            yield chunk


