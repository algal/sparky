"""
Unit tests for OpenClaw Gateway WebSocket client.

Tests the protocol handling: handshake, cumulative text → incremental
conversion, sparse delta tail extraction, multi-step tool use, termination
logic, and error handling. Uses a mock WebSocket.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
import websockets.exceptions
from websockets.frames import Close

from sparky_mvp.core.openclaw_gateway_client import (
    OpenClawGatewayClient,
    OpenClawEvent,
    OpenClawEventType,
    create_gateway_client_from_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_event(state: str, text: str, content_type: str = "text",
                run_id: str = "run-123") -> str:
    """Build a JSON chat event frame.

    Includes runId in the payload to match the real Gateway protocol.
    Default run_id matches the default in _ack_response().
    """
    return json.dumps({
        "type": "event",
        "event": "chat",
        "payload": {
            "runId": run_id,
            "state": state,
            "message": {
                "content": [{"type": content_type, "text": text}],
            },
        },
    })


def _agent_event(stream: str, data: dict, run_id: str = "run-123") -> str:
    """Build a JSON agent event frame.

    Includes runId in the payload to match the real Gateway protocol.
    """
    return json.dumps({
        "type": "event",
        "event": "agent",
        "payload": {"runId": run_id, "stream": stream, "data": data},
    })


def _ack_response(req_id: str, ok: bool = True, run_id: str = "run-123") -> str:
    """Build a chat.send ACK response."""
    frame = {"type": "res", "id": req_id, "ok": ok}
    if ok:
        frame["payload"] = {"runId": run_id}
    else:
        frame["error"] = {"message": "something went wrong"}
    return json.dumps(frame)


def _connect_challenge() -> str:
    return json.dumps({"event": "connect.challenge", "payload": {"nonce": "abc"}})


def _connect_response(req_id: str, ok: bool = True) -> str:
    frame = {"type": "res", "id": req_id, "ok": ok}
    if ok:
        frame["payload"] = {"protocol": 3, "server": {"version": "1.0.0"}}
    else:
        frame["error"] = {"message": "auth failed"}
    return json.dumps(frame)


class MockWebSocket:
    """Simulates a WebSocket connection with a sequence of frames."""

    def __init__(self, frames: list[object]):
        self._frames = list(frames)
        self._idx = 0
        self._sent: list[str] = []
        self.close_code: int | None = None  # None = open, int = closed

    async def recv(self) -> str:
        if self._idx >= len(self._frames):
            # Simulate timeout by waiting forever (caller will timeout)
            await asyncio.sleep(999)
        frame = self._frames[self._idx]
        self._idx += 1
        if isinstance(frame, Exception):
            raise frame
        if frame == "__TIMEOUT__":
            await asyncio.sleep(999)
        return frame

    async def send(self, data: str) -> None:
        self._sent.append(data)

    async def close(self) -> None:
        pass


async def _collect_events(client, session_key="test", message="hello") -> list[OpenClawEvent]:
    """Collect all events from chat_send into a list."""
    events = []
    async for event in client.chat_send(session_key=session_key, message=message):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Handshake tests
# ---------------------------------------------------------------------------


class TestConnect:

    @pytest.mark.asyncio
    async def test_successful_handshake(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("r1"),
        ])
        with patch("sparky_mvp.core.openclaw_gateway_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawGatewayClient(token="secret")
            await client.connect()

            assert client.is_connected
            assert client._protocol_version == 3
            # Verify connect request was sent with correct params
            sent = json.loads(ws._sent[0])
            assert sent["method"] == "connect"
            assert sent["params"]["client"]["id"] == "gateway-client"
            assert sent["params"]["client"]["mode"] == "backend"
            assert sent["params"]["auth"]["token"] == "secret"

    @pytest.mark.asyncio
    async def test_bad_challenge_raises(self):
        ws = MockWebSocket([
            json.dumps({"event": "not.a.challenge"}),
        ])
        with patch("sparky_mvp.core.openclaw_gateway_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawGatewayClient()
            with pytest.raises(ConnectionError, match="Expected connect.challenge"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_rejected_connect_raises(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("r1", ok=False),
        ])
        with patch("sparky_mvp.core.openclaw_gateway_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawGatewayClient()
            with pytest.raises(ConnectionError, match="auth failed"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_empty_token_omits_auth(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("r1"),
        ])
        with patch("sparky_mvp.core.openclaw_gateway_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawGatewayClient(token="")
            await client.connect()

            sent = json.loads(ws._sent[0])
            assert sent["params"]["auth"] == {}

    @pytest.mark.asyncio
    async def test_idempotent_connect(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("r1"),
        ])
        with patch("sparky_mvp.core.openclaw_gateway_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawGatewayClient()
            await client.connect()
            # Second connect should be no-op
            await client.connect()
            assert mock_ws.connect.call_count == 1


# ---------------------------------------------------------------------------
# chat_send: basic flow
# ---------------------------------------------------------------------------


class TestChatSendBasic:

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        """Create a client that's already 'connected' and return (client, ws)."""
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0  # so first chat req_id = "r1"
        return client, ws

    @pytest.mark.asyncio
    async def test_simple_delta_then_final(self):
        """Single delta followed by final — basic happy path."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _chat_event("delta", "Hello world"),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Hello world"),
        ])

        events = await _collect_events(client)

        types = [e.type for e in events]
        assert OpenClawEventType.CHAT_DELTA in types
        assert OpenClawEventType.CHAT_FINAL in types

        # Delta should contain full incremental text
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]
        assert len(deltas) == 1
        assert deltas[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_cumulative_deltas_converted_to_incremental(self):
        """Multiple cumulative deltas should yield only the new portions."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _chat_event("delta", "Hel"),
            _chat_event("delta", "Hello"),
            _chat_event("delta", "Hello world"),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Hello world"),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        # Should get 3 incremental pieces
        assert len(deltas) == 3
        assert deltas[0].text == "Hel"
        assert deltas[1].text == "lo"
        assert deltas[2].text == " world"

    @pytest.mark.asyncio
    async def test_final_tail_extraction(self):
        """chat_final with text beyond last delta yields a tail CHAT_DELTA."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _chat_event("delta", "Yes"),
            # Final has more text than the last delta
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Yes, I can hear you."),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        assert len(deltas) == 2
        assert deltas[0].text == "Yes"
        assert deltas[1].text == ", I can hear you."

    @pytest.mark.asyncio
    async def test_no_deltas_only_final(self):
        """Edge case: no deltas at all, only final with text."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Direct answer"),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        assert len(deltas) == 1
        assert deltas[0].text == "Direct answer"

    @pytest.mark.asyncio
    async def test_empty_final_no_tail(self):
        """Final with no text should not yield a tail delta."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", ""),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]
        assert len(deltas) == 0

    @pytest.mark.asyncio
    async def test_final_text_matches_last_delta_no_tail(self):
        """If final text == last delta text, no extra tail delta."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _chat_event("delta", "Exact match"),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Exact match"),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        # Only the one delta from the delta event
        assert len(deltas) == 1
        assert deltas[0].text == "Exact match"


# ---------------------------------------------------------------------------
# chat_send: multi-step tool use
# ---------------------------------------------------------------------------


class TestChatSendToolUse:

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_cumulative_text_reset_after_tool(self):
        """After tool use, cumulative text resets for the next LLM step."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("lifecycle", {"phase": "start"}),
            # First LLM step: "Let me check..."
            _chat_event("delta", "Let me check"),
            _chat_event("final", "Let me check the time."),
            # Tool call
            _agent_event("tool", {"name": "get_time", "args": {}}),
            # Second LLM step: cumulative text resets (shorter than step 1)
            _chat_event("delta", "It's 3pm"),
            _chat_event("final", "It's 3pm."),
            _agent_event("lifecycle", {"phase": "end"}),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        # Step 1: "Let me check" + tail " the time."
        # Step 2: "It's 3pm" + tail "."
        assert len(deltas) == 4
        assert deltas[0].text == "Let me check"
        assert deltas[1].text == " the time."
        assert deltas[2].text == "It's 3pm"
        assert deltas[3].text == "."

    @pytest.mark.asyncio
    async def test_tool_trace_event(self):
        """Tool events are yielded as TOOL_TRACE."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("tool", {"name": "get_time", "args": {"tz": "PST"}}),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "It's 6pm."),
        ])

        events = await _collect_events(client)
        tool_events = [e for e in events if e.type == OpenClawEventType.TOOL_TRACE]

        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "get_time"
        assert tool_events[0].tool_args == {"tz": "PST"}


# ---------------------------------------------------------------------------
# chat_send: lifecycle events
# ---------------------------------------------------------------------------


class TestChatSendLifecycle:

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_lifecycle_end_does_not_terminate(self):
        """lifecycle:end is informational — must NOT terminate the stream."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("lifecycle", {"phase": "start"}),
            _chat_event("delta", "Hello"),
            _agent_event("lifecycle", {"phase": "end"}),
            # lifecycle:end arrived but chat_final hasn't — must keep going
            _chat_event("final", "Hello world"),
        ])

        events = await _collect_events(client)

        # Should have lifecycle events
        assert any(e.type == OpenClawEventType.LIFECYCLE_START for e in events)
        assert any(e.type == OpenClawEventType.LIFECYCLE_END for e in events)

        # Must have received chat_final (proves we didn't terminate early)
        assert any(e.type == OpenClawEventType.CHAT_FINAL for e in events)

        # Must have the tail text from final
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]
        full = "".join(d.text for d in deltas)
        assert full == "Hello world"

    @pytest.mark.asyncio
    async def test_assistant_stream_ignored(self):
        """event:agent stream=assistant should not produce events (redundant)."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            _agent_event("assistant", {"text": "Hello"}),
            _agent_event("assistant", {"text": "Hello world"}),
            _agent_event("lifecycle", {"phase": "end"}),
            _chat_event("final", "Hello world"),
        ])

        events = await _collect_events(client)

        # No assistant events should appear — they're filtered
        types = {e.type for e in events}
        assert OpenClawEventType.CHAT_DELTA in types  # from final tail
        assert OpenClawEventType.CHAT_FINAL in types
        # No lifecycle or tool events
        assert OpenClawEventType.LIFECYCLE_START not in types
        assert OpenClawEventType.TOOL_TRACE not in types


# ---------------------------------------------------------------------------
# chat_send: error handling
# ---------------------------------------------------------------------------


class TestChatSendErrors:

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_rejected_chat_send(self):
        """chat.send rejection yields ERROR and stops."""
        client, ws = self._make_connected_client([
            _ack_response("r1", ok=False),
        ])

        events = await _collect_events(client)

        assert len(events) == 1
        assert events[0].type == OpenClawEventType.ERROR
        assert "rejected" in events[0].error_message

    @pytest.mark.asyncio
    async def test_timeout_yields_error(self):
        """Timeout during recv yields ERROR event."""
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            "__TIMEOUT__",  # will cause asyncio.sleep(999) -> timeout
        ])

        events = []
        # Use a very short timeout to trigger the timeout path
        async for event in client.chat_send(
            session_key="test",
            message="hello",
            timeout_ms=100,  # 0.1s timeout
        ):
            events.append(event)

        assert any(e.type == OpenClawEventType.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_lifecycle_then_close_pre_text_retries_and_dedupes(self):
        """If close happens before any text, retry should recover without duplicate lifecycle."""
        first_ws = MockWebSocket([
            _ack_response("r1", run_id="run-123"),
            _agent_event("lifecycle", {"phase": "start"}, run_id="run-123"),
            websockets.exceptions.ConnectionClosedError(
                Close(1011, "keepalive timeout"),
                None,
            ),
        ])
        second_ws = MockWebSocket([
            _ack_response("r2", run_id="run-123"),
            # Replayed by server after retry; should be de-duped client-side
            _agent_event("lifecycle", {"phase": "start"}, run_id="run-123"),
            _agent_event("lifecycle", {"phase": "end"}, run_id="run-123"),
            _chat_event("final", "Done", run_id="run-123"),
        ])

        client = OpenClawGatewayClient()
        client._ws = first_ws
        client._connected = True
        client._req_counter = 0

        async def _reconnect():
            client._ws = second_ws
            client._connected = True

        client.connect = AsyncMock(side_effect=_reconnect)

        events = await _collect_events(client)
        lifecycle_starts = [
            e for e in events if e.type == OpenClawEventType.LIFECYCLE_START
        ]
        assert client.connect.call_count == 1
        assert len(lifecycle_starts) == 1
        assert any(e.type == OpenClawEventType.CHAT_FINAL for e in events)

    @pytest.mark.asyncio
    async def test_recv_pre_yield_retries_on_connection_closed(self):
        """Before any yields, ConnectionClosed triggers reconnect + retry."""
        first_ws = MockWebSocket([
            _ack_response("r1", run_id="run-123"),
            websockets.exceptions.ConnectionClosedError(
                Close(1011, "keepalive timeout"),
                None,
            ),
        ])
        second_ws = MockWebSocket([
            _ack_response("r2", run_id="run-123"),
            _agent_event("lifecycle", {"phase": "end"}, run_id="run-123"),
            _chat_event("final", "Done", run_id="run-123"),
        ])

        client = OpenClawGatewayClient()
        client._ws = first_ws
        client._connected = True
        client._req_counter = 0

        async def _reconnect():
            client._ws = second_ws
            client._connected = True

        client.connect = AsyncMock(side_effect=_reconnect)

        events = await _collect_events(client)
        assert client.connect.call_count == 1
        assert any(e.type == OpenClawEventType.CHAT_FINAL for e in events)

    @pytest.mark.asyncio
    async def test_close_after_text_raises_without_retry(self):
        """After text is yielded, ConnectionClosed should propagate (no replay retry)."""
        ws = MockWebSocket([
            _ack_response("r1", run_id="run-123"),
            _chat_event("delta", "Hel", run_id="run-123"),
            websockets.exceptions.ConnectionClosedError(
                Close(1011, "keepalive timeout"),
                None,
            ),
        ])

        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        client.connect = AsyncMock()

        yielded = []
        with pytest.raises(websockets.exceptions.ConnectionClosedError):
            async for event in client.chat_send(session_key="test", message="hello"):
                yielded.append(event)

        deltas = [e for e in yielded if e.type == OpenClawEventType.CHAT_DELTA]
        assert len(deltas) == 1
        assert deltas[0].text == "Hel"
        assert client.connect.call_count == 0

    @pytest.mark.asyncio
    async def test_retry_preserves_identical_tool_events(self):
        """Identical tool events after retry are legitimate and must not be collapsed."""
        first_ws = MockWebSocket([
            _ack_response("r1", run_id="run-123"),
            websockets.exceptions.ConnectionClosedError(
                Close(1011, "keepalive timeout"),
                None,
            ),
        ])
        second_ws = MockWebSocket([
            _ack_response("r2", run_id="run-123"),
            _agent_event("tool", {"name": "camera.snap", "args": {"camera": "front"}}, run_id="run-123"),
            _agent_event("tool", {"name": "camera.snap", "args": {"camera": "front"}}, run_id="run-123"),
            _agent_event("lifecycle", {"phase": "end"}, run_id="run-123"),
            _chat_event("final", "Done", run_id="run-123"),
        ])

        client = OpenClawGatewayClient()
        client._ws = first_ws
        client._connected = True
        client._req_counter = 0

        async def _reconnect():
            client._ws = second_ws
            client._connected = True

        client.connect = AsyncMock(side_effect=_reconnect)

        events = await _collect_events(client)
        tool_events = [e for e in events if e.type == OpenClawEventType.TOOL_TRACE]

        assert client.connect.call_count == 1
        assert len(tool_events) == 2


# ---------------------------------------------------------------------------
# _extract_text helper
# ---------------------------------------------------------------------------


class TestExtractText:

    def test_dict_blocks(self):
        payload = {
            "message": {
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}],
            },
        }
        assert OpenClawGatewayClient._extract_text(payload) == "Hello world"

    def test_string_blocks(self):
        payload = {
            "message": {
                "content": ["Hello", " world"],
            },
        }
        assert OpenClawGatewayClient._extract_text(payload) == "Hello world"

    def test_mixed_blocks(self):
        payload = {
            "message": {
                "content": [{"type": "text", "text": "Hello"}, " world"],
            },
        }
        assert OpenClawGatewayClient._extract_text(payload) == "Hello world"

    def test_non_text_blocks_skipped(self):
        payload = {
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "url": "..."},
                ],
            },
        }
        assert OpenClawGatewayClient._extract_text(payload) == "Hello"

    def test_empty_content(self):
        payload = {"message": {"content": []}}
        assert OpenClawGatewayClient._extract_text(payload) == ""

    def test_missing_message(self):
        payload = {}
        assert OpenClawGatewayClient._extract_text(payload) == ""


# ---------------------------------------------------------------------------
# run_id extraction
# ---------------------------------------------------------------------------


class TestRunIdExtraction:

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_run_id_from_ack(self):
        """run_id should come from ACK payload."""
        client, ws = self._make_connected_client([
            _ack_response("r1", run_id="custom-run-42"),
            _agent_event("lifecycle", {"phase": "end"}, run_id="custom-run-42"),
            _chat_event("final", "Done", run_id="custom-run-42"),
        ])

        events = await _collect_events(client)
        final = [e for e in events if e.type == OpenClawEventType.CHAT_FINAL][0]
        assert final.run_id == "custom-run-42"

    @pytest.mark.asyncio
    async def test_run_id_fallback_to_idempotency_key(self):
        """If ACK has no runId, fallback to idempotency key."""
        ack = json.dumps({"type": "res", "id": "r1", "ok": True, "payload": {}})
        client, ws = self._make_connected_client([
            ack,
            # run_id must match the idempotency key fallback ("turn_my-turn")
            _agent_event("lifecycle", {"phase": "end"}, run_id="turn_my-turn"),
            _chat_event("final", "Done", run_id="turn_my-turn"),
        ])

        events = []
        async for event in client.chat_send(
            session_key="test", message="hello", turn_id="my-turn"
        ):
            events.append(event)

        final = [e for e in events if e.type == OpenClawEventType.CHAT_FINAL][0]
        assert final.run_id == "turn_my-turn"

    @pytest.mark.asyncio
    async def test_stale_events_from_wrong_run_skipped(self):
        """Events with a different runId (from a previous aborted turn)
        should be silently skipped, not mistaken for the current turn."""
        client, ws = self._make_connected_client([
            # Stale events from a PREVIOUS turn (wrong runId)
            _chat_event("delta", "I am not Siri", run_id="old-aborted-run"),
            _chat_event("final", "I am not Siri, I am Dobby", run_id="old-aborted-run"),
            # Our turn's events (correct runId)
            _ack_response("r1", run_id="current-run"),
            _chat_event("delta", "Correct answer", run_id="current-run"),
            _agent_event("lifecycle", {"phase": "end"}, run_id="current-run"),
            _chat_event("final", "Correct answer", run_id="current-run"),
        ])

        events = await _collect_events(client)
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]

        # Should only see the current turn's text, NOT the stale "Siri" text
        assert len(deltas) == 1
        assert deltas[0].text == "Correct answer"
        assert deltas[0].run_id == "current-run"

    @pytest.mark.asyncio
    async def test_aborted_state_terminates(self):
        """Server-sent state='aborted' should yield ERROR and stop."""
        aborted_event = json.dumps({
            "type": "event",
            "event": "chat",
            "payload": {
                "runId": "run-123",
                "state": "aborted",
                "stopReason": "user_abort",
            },
        })
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            aborted_event,
        ])

        events = await _collect_events(client)
        assert len(events) == 1
        assert events[0].type == OpenClawEventType.ERROR
        assert "aborted" in events[0].error_message


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:

    def test_create_from_config_basic(self):
        config = {
            "openclaw": {
                "gateway_ws_url": "ws://custom:9999",
                "gateway_token_env": "MY_TOKEN_VAR",
            },
        }
        with patch.dict("os.environ", {"MY_TOKEN_VAR": "test-token"}):
            client = create_gateway_client_from_config(config)
            assert client.gateway_url == "ws://custom:9999"
            assert client.token == "test-token"

    def test_create_from_config_defaults(self):
        config = {}
        with patch.dict("os.environ", {}, clear=True):
            client = create_gateway_client_from_config(config)
            assert client.gateway_url == "ws://127.0.0.1:18789"


# ---------------------------------------------------------------------------
# chat.inject
# ---------------------------------------------------------------------------


class TestChatInject:

    def _make_connected_client(self, response_frames: list[str]) -> tuple:
        ws = MockWebSocket(response_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_inject_success(self):
        """chat.inject returns True on ok response."""
        res = json.dumps({"type": "res", "id": "r1", "ok": True, "payload": {"messageId": "m1"}})
        client, ws = self._make_connected_client([res])

        result = await client.chat_inject("agent:main:reachy", "Hello", label="TEST")
        assert result is True

        # Verify the sent request shape
        sent = json.loads(ws._sent[0])
        assert sent["method"] == "chat.inject"
        assert sent["params"]["sessionKey"] == "agent:main:reachy"
        assert sent["params"]["message"] == "Hello"
        assert sent["params"]["label"] == "TEST"

    @pytest.mark.asyncio
    async def test_inject_without_label(self):
        """chat.inject omits label when not provided."""
        res = json.dumps({"type": "res", "id": "r1", "ok": True, "payload": {}})
        client, ws = self._make_connected_client([res])

        await client.chat_inject("test-session", "Voice mode on")
        sent = json.loads(ws._sent[0])
        assert "label" not in sent["params"]

    @pytest.mark.asyncio
    async def test_inject_failure(self):
        """chat.inject returns False on error response."""
        res = json.dumps({"type": "res", "id": "r1", "ok": False, "error": {"message": "not found"}})
        client, ws = self._make_connected_client([res])

        result = await client.chat_inject("bad-session", "Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_inject_wrong_response_id_returns_false(self):
        """chat.inject must not accept a response for a different request id."""
        res = json.dumps({"type": "res", "id": "wrong-id", "ok": True, "payload": {}})
        client, ws = self._make_connected_client([res])

        result = await client.chat_inject("agent:main:reachy", "Hello", timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_inject_skips_unrelated_frames(self):
        """chat.inject must skip unrelated frames (health ticks, etc.) and find its ACK."""
        health_tick = json.dumps({"type": "event", "event": "health", "payload": {}})
        stale_event = json.dumps({"type": "event", "event": "chat", "payload": {"runId": "old"}})
        correct_ack = json.dumps({"type": "res", "id": "r1", "ok": True, "payload": {}})
        client, ws = self._make_connected_client([health_tick, stale_event, correct_ack])

        result = await client.chat_inject("agent:main:reachy", "Hello")
        assert result is True


# ---------------------------------------------------------------------------
# Bug-exposing tests (failing tests that document known bugs)
# ---------------------------------------------------------------------------


class TestKnownBugs:
    """Tests that expose confirmed bugs.

    Each test documents a real bug found during code review. The test
    SHOULD pass when the bug is fixed. Until then, these tests are
    expected to fail (marked xfail).
    """

    def _make_connected_client(self, chat_frames: list[str]) -> tuple:
        ws = MockWebSocket(chat_frames)
        client = OpenClawGatewayClient()
        client._ws = ws
        client._connected = True
        client._req_counter = 0
        return client, ws

    @pytest.mark.asyncio
    async def test_multistep_tool_use_receives_all_steps(self):
        """Multi-step tool-use turn: LLM speaks, calls tool, then speaks again.

        The Gateway sends two chat_final events — one per LLM step. The
        client must NOT terminate on the first chat_final; it should keep
        reading until both lifecycle:end and a chat_final have been seen.
        """
        client, ws = self._make_connected_client([
            _ack_response("r1"),
            # LLM step 1: preamble
            _agent_event("lifecycle", {"phase": "start"}),
            _chat_event("delta", "Let me check"),
            _chat_event("final", "Let me check the time."),
            # Tool call
            _agent_event("tool", {"name": "get_time", "args": {}}),
            # LLM step 2: actual answer after tool result
            _chat_event("delta", "It's 3:42 PM"),
            _chat_event("final", "It's 3:42 PM!"),
            _agent_event("lifecycle", {"phase": "end"}),
        ])

        events = await _collect_events(client)

        # We should see BOTH text segments
        deltas = [e for e in events if e.type == OpenClawEventType.CHAT_DELTA]
        full_text = "".join(d.text for d in deltas)
        assert "3:42" in full_text, (
            f"Second LLM step (the actual answer) was lost. Got: {full_text!r}"
        )

        # We should see the tool trace
        tool_events = [e for e in events if e.type == OpenClawEventType.TOOL_TRACE]
        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "get_time"

    @pytest.mark.asyncio
    async def test_recv_retry_has_bounded_attempts(self):
        """If the WS keeps dying before text is yielded, retry must give up.

        The recv-path retry (ConnectionClosed before _yielded_text) must
        have an attempt limit. If the Gateway keeps crashing immediately
        after ACK, chat_send should yield an ERROR event after exhausting
        retries, not loop forever.
        """
        # Create a sequence of WebSockets that always crash after ACK.
        crash_error = websockets.exceptions.ConnectionClosedError(
            Close(1011, "keepalive timeout"), None,
        )
        ws_sequence = []
        for i in range(5):  # more than the retry limit (3)
            ws_sequence.append(MockWebSocket([
                _ack_response(f"r{i + 1}", run_id="run-123"),
                crash_error,
            ]))

        ws_iter = iter(ws_sequence)
        client = OpenClawGatewayClient()
        client._ws = next(ws_iter)
        client._connected = True
        client._req_counter = 0

        async def _reconnect():
            client._ws = next(ws_iter)
            client._connected = True

        client.connect = AsyncMock(side_effect=_reconnect)

        # chat_send must terminate (not hang) and yield an error.
        events = await _collect_events(client)

        error_events = [e for e in events if e.type == OpenClawEventType.ERROR]
        assert len(error_events) == 1
        assert "retries" in error_events[0].error_message

        # Should have retried fewer than 5 times (the limit is 3).
        assert client.connect.call_count <= 3
