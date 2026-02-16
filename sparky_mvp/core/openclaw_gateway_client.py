"""
Async WebSocket client for OpenClaw Gateway.

Maintains a persistent connection and exposes chat_send() as an async iterator
of typed events. Designed for the Reachy bridge: connect once, send turns,
stream deltas.

Protocol: OpenClaw Gateway WS v3 (JSON over WebSocket).
Auth: token-based (no device identity required on localhost).
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import websockets
import websockets.exceptions
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types returned by chat_send()
# ---------------------------------------------------------------------------

class OpenClawEventType(Enum):
    CHAT_DELTA = "chat_delta"
    CHAT_FINAL = "chat_final"
    LIFECYCLE_START = "lifecycle_start"
    LIFECYCLE_END = "lifecycle_end"
    TOOL_TRACE = "tool_trace"
    ERROR = "error"


@dataclass
class OpenClawEvent:
    type: OpenClawEventType
    text: str = ""
    run_id: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    error_message: str = ""
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenClawGatewayClient:
    """
    Persistent async WS client for the OpenClaw Gateway.

    Usage::

        client = OpenClawGatewayClient(gateway_url, token)
        await client.connect()

        async for event in client.chat_send("reachy", "Hello!"):
            if event.type == OpenClawEventType.CHAT_DELTA:
                print(event.text, end="", flush=True)

        await client.close()
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789",
        token: str = "",
        client_id: str = "gateway-client",
        client_mode: str = "backend",
    ):
        self.gateway_url = gateway_url
        self.token = token
        self.client_id = client_id
        self.client_mode = client_mode
        self._ws: Optional[ClientConnection] = None
        self._connected = False
        self._protocol_version: Optional[int] = None
        self._req_counter = 0
        self.last_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Gateway and complete the protocol v3 handshake."""
        if self._connected and self._ws:
            logger.debug("Already connected")
            return

        logger.info("Connecting to Gateway at %s ...", self.gateway_url)
        self._ws = await websockets.connect(self.gateway_url)

        # 1. Receive connect.challenge
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        challenge = json.loads(raw)
        if challenge.get("event") != "connect.challenge":
            raise ConnectionError(
                f"Expected connect.challenge, got: {challenge.get('event')}"
            )
        logger.debug("Got connect.challenge")

        # 2. Send connect request
        req_id = self._next_req_id()
        connect_req = {
            "type": "req",
            "id": req_id,
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": self.client_id,
                    "version": "0.1.0",
                    "platform": "linux",
                    "mode": self.client_mode,
                },
                "role": "operator",
                "scopes": ["operator.read", "operator.write"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self.token} if self.token else {},
                "locale": "en-US",
                "userAgent": "reachy-bridge/0.1.0",
            },
        }
        await self._ws.send(json.dumps(connect_req))

        # 3. Wait for connect response
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        res = json.loads(raw)
        if res.get("type") != "res" or res.get("id") != req_id:
            raise ConnectionError(f"Unexpected connect response: {res}")
        if not res.get("ok"):
            error = res.get("error", {})
            raise ConnectionError(
                f"Gateway rejected connect: {error.get('message', error)}"
            )

        payload = res.get("payload", {})
        self._protocol_version = payload.get("protocol")
        self._connected = True
        logger.info(
            "Connected to Gateway (protocol=%s, server=%s)",
            self._protocol_version,
            payload.get("server", {}).get("version", "?"),
        )

        # 4. Drain initial events (health, presence, etc.)
        drained = 0
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=0.3)
                drained += 1
            except asyncio.TimeoutError:
                break
        if drained:
            logger.debug("Drained %d initial events", drained)

    async def ensure_connected(self) -> None:
        """Connect if not already connected, or reconnect if the WS died."""
        if self._connected and self._ws is not None:
            # Check actual WS state — keepalive timeout closes the connection
            # but our flags stay set.  close_code is None while open.
            if self._ws.close_code is None:
                return  # Still alive
            logger.warning(
                "Gateway WS connection died (close_code=%s) — reconnecting...",
                self._ws.close_code,
            )
            self._ws = None
            self._connected = False
        await self.connect()

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False
        logger.info("Gateway connection closed")

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._ws is not None
            and self._ws.close_code is None
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat_send(
        self,
        session_key: str,
        message: str,
        timeout_ms: int = 60000,
        turn_id: Optional[str] = None,
    ) -> AsyncIterator[OpenClawEvent]:
        """
        Send a chat message and yield streaming events.

        Args:
            session_key: OpenClaw session key (e.g. "reachy")
            message: The user message text
            timeout_ms: Server-side timeout for the agent run
            turn_id: Optional stable turn ID (for idempotency). Auto-generated if omitted.

        Yields:
            OpenClawEvent instances (CHAT_DELTA, CHAT_FINAL, LIFECYCLE_*, TOOL_TRACE, ERROR)
        """
        await self.ensure_connected()

        if turn_id is None:
            turn_id = str(uuid.uuid4())
        idempotency_key = f"turn_{turn_id}"
        req_id = self._next_req_id()

        chat_req = {
            "type": "req",
            "id": req_id,
            "method": "chat.send",
            "params": {
                "sessionKey": session_key,
                "message": message,
                "idempotencyKey": idempotency_key,
                "timeoutMs": timeout_ms,
            },
        }

        logger.info(
            "chat.send: session=%s idempotencyKey=%s msg=%s",
            session_key,
            idempotency_key,
            message[:80],
        )

        # Send with one retry on dead connection.  The idempotencyKey is
        # stable across retries so the Gateway deduplicates if the first
        # attempt was partially received.
        for _attempt in range(2):
            try:
                assert self._ws is not None
                await self._ws.send(json.dumps(chat_req))
                break
            except websockets.exceptions.ConnectionClosed as exc:
                if _attempt == 0:
                    logger.warning(
                        "WS dead on send — reconnecting and retrying: %s", exc,
                    )
                    self._ws = None
                    self._connected = False
                    await self.ensure_connected()
                    # Need a fresh req_id for the new connection; keep
                    # the same idempotencyKey for dedup.
                    req_id = self._next_req_id()
                    chat_req["id"] = req_id
                    continue
                raise

        # Collect events until the run completes.
        #
        # Termination: a turn is done when we've seen BOTH lifecycle:end AND
        # at least one chat_final.  The ordering varies:
        #   - Single-step turns: lifecycle:end → chat_final
        #   - Multi-step (tool use): chat_final(step1) → tool → chat_final(step2) → lifecycle:end
        # After each chat_final we reset _cumulative_text_len so the next LLM
        # step's cumulative text is handled correctly.
        got_ack = False
        run_id = idempotency_key
        done = False
        _got_lifecycle_end = False
        _got_any_chat_final = False
        # event:chat deltas are CUMULATIVE (full text so far). Track what
        # we've already yielded so we can emit only the incremental part.
        _cumulative_text_len = 0
        # Retry safety gate:
        # - Before any text is yielded, reconnect+retry is safe.
        # - After any text is yielded, do NOT retry (caller consumed partial output).
        _yielded_text = False
        # If we retry after yielding non-text events (lifecycle/tool), the
        # server may replay those events. Use occurrence counts rather than
        # set-membership so legitimate repeated identical events still pass.
        _recv_retry_count = 0
        _non_text_emitted_counts: dict[tuple[str, str, str], int] = {}
        _replay_skip_budget: dict[tuple[str, str, str], int] = {}

        while not done:
            try:
                assert self._ws is not None
                raw = await asyncio.wait_for(
                    self._ws.recv(), timeout=timeout_ms / 1000 + 10
                )
            except websockets.exceptions.ConnectionClosed as exc:
                if not _yielded_text:
                    _recv_retry_count += 1
                    if _recv_retry_count >= 3:
                        logger.error(
                            "WS recv retry limit reached (%d attempts) — giving up: %s",
                            _recv_retry_count, exc,
                        )
                        yield OpenClawEvent(
                            type=OpenClawEventType.ERROR,
                            error_message=(
                                f"Gateway connection failed after {_recv_retry_count} retries"
                            ),
                        )
                        return
                    logger.warning(
                        "WS dead on recv (pre-data, attempt %d/3) — reconnecting and retrying: %s",
                        _recv_retry_count, exc,
                    )
                    self._ws = None
                    self._connected = False
                    await self.ensure_connected()
                    _replay_skip_budget = dict(_non_text_emitted_counts)
                    req_id = self._next_req_id()
                    chat_req["id"] = req_id
                    assert self._ws is not None
                    await self._ws.send(json.dumps(chat_req))
                    got_ack = False
                    run_id = idempotency_key
                    _cumulative_text_len = 0
                    _got_lifecycle_end = False
                    _got_any_chat_final = False
                    continue
                raise
            except asyncio.TimeoutError:
                yield OpenClawEvent(
                    type=OpenClawEventType.ERROR,
                    error_message=f"Timeout waiting for Gateway response ({timeout_ms}ms)",
                )
                return

            frame = json.loads(raw)
            ftype = frame.get("type")
            event_name = frame.get("event", "")

            # ── Per-frame debug trace ──
            # Log every frame received during chat_send so we can diagnose
            # event ordering issues, stale events, and premature termination.
            _frame_run_id = ""
            _frame_summary = f"type={ftype}"
            if ftype == "event":
                payload_peek = frame.get("payload", {})
                _frame_run_id = payload_peek.get("runId", "")
                if event_name == "chat":
                    _frame_summary = (
                        f"event:chat state={payload_peek.get('state', '?')}"
                        f" runId={_frame_run_id}"
                        f" textLen={len(self._extract_text(payload_peek))}"
                    )
                elif event_name == "agent":
                    stream = payload_peek.get("stream", "?")
                    phase = payload_peek.get("data", {}).get("phase", "")
                    _frame_summary = (
                        f"event:agent stream={stream}"
                        f" phase={phase}"
                        f" runId={_frame_run_id}"
                    )
                else:
                    _frame_summary = f"event:{event_name}"
            elif ftype == "res":
                _frame_summary = (
                    f"res id={frame.get('id')} ok={frame.get('ok')}"
                )
            logger.debug(
                "[ws-frame] %s | got_ack=%s our_runId=%s",
                _frame_summary, got_ack, run_id,
            )

            # --- Handle chat.send ACK ---
            if ftype == "res" and frame.get("id") == req_id:
                got_ack = True
                if not frame.get("ok"):
                    error = frame.get("error", {})
                    logger.debug("[ws-decision] ACK error — terminating")
                    yield OpenClawEvent(
                        type=OpenClawEventType.ERROR,
                        error_message=f"chat.send rejected: {error.get('message', error)}",
                        raw=frame,
                    )
                    return
                ack_payload = frame.get("payload", {})
                run_id = ack_payload.get("runId", idempotency_key)
                self.last_run_id = run_id
                logger.debug("[ws-decision] ACK accepted: runId=%s", run_id)
                continue

            # --- Handle event:chat (delta / final / aborted) ---
            if ftype == "event" and event_name == "chat":
                payload = frame.get("payload", {})

                # ── Run-ID filtering ──
                # Every event:chat frame carries payload.runId identifying
                # which agent run (turn) it belongs to.  After a barge-in
                # cancellation, stale events from the aborted run may still
                # arrive on this shared WS.  Skip any that don't match ours.
                event_run_id = payload.get("runId", "")
                if not got_ack:
                    # Before our ACK, any chat event is leftover from a
                    # prior turn — skip it.
                    logger.debug(
                        "[ws-decision] SKIP pre-ACK chat event: runId=%s",
                        event_run_id,
                    )
                    continue
                if event_run_id and event_run_id != run_id:
                    logger.debug(
                        "[ws-decision] SKIP chat event wrong runId: got=%s want=%s",
                        event_run_id, run_id,
                    )
                    continue

                state = payload.get("state", "")

                # Server-side abort: the Gateway sends state="aborted" when
                # chat.abort successfully cancelled the run.  Treat as
                # terminal — no more events will follow for this run.
                if state == "aborted":
                    logger.warning(
                        "[ws-decision] TERMINAL: chat aborted runId=%s", run_id,
                    )
                    yield OpenClawEvent(
                        type=OpenClawEventType.ERROR,
                        error_message="Run aborted by server",
                        run_id=run_id,
                        raw=frame,
                    )
                    return

                text = self._extract_text(payload)

                if state == "delta" and text:
                    # Deltas are cumulative — only yield the new portion.
                    # Handle cumulative text reset (new LLM step after tool use):
                    if len(text) < _cumulative_text_len:
                        logger.debug(
                            "Cumulative text reset: %d -> %d (new LLM step)",
                            _cumulative_text_len, len(text),
                        )
                        _cumulative_text_len = 0
                    new_text = text[_cumulative_text_len:]
                    _cumulative_text_len = len(text)
                    if new_text:
                        logger.debug(
                            "[ws-decision] YIELD chat_delta: +%d chars (cumulative=%d)",
                            len(new_text), _cumulative_text_len,
                        )
                        _yielded_text = True
                        _non_text_emitted_counts.clear()
                        _replay_skip_budget.clear()
                        yield OpenClawEvent(
                            type=OpenClawEventType.CHAT_DELTA,
                            text=new_text,
                            run_id=run_id,
                            raw=frame,
                        )
                elif state == "final":
                    # chat_final contains the complete cumulative text for this
                    # LLM step.  event:chat deltas are sparse (batched), so the
                    # final often has text beyond the last delta.  Yield any
                    # remaining portion before emitting the CHAT_FINAL event.
                    if text and len(text) > _cumulative_text_len:
                        tail = text[_cumulative_text_len:]
                        _cumulative_text_len = len(text)
                        _yielded_text = True
                        _non_text_emitted_counts.clear()
                        _replay_skip_budget.clear()
                        yield OpenClawEvent(
                            type=OpenClawEventType.CHAT_DELTA,
                            text=tail,
                            run_id=run_id,
                            raw=frame,
                        )
                    yield OpenClawEvent(
                        type=OpenClawEventType.CHAT_FINAL,
                        text=text,
                        run_id=run_id,
                        raw=frame,
                    )
                    _got_any_chat_final = True
                    # Reset cumulative text for the next LLM step (if any).
                    _cumulative_text_len = 0
                    # Terminate only if we've also seen lifecycle:end.
                    if _got_lifecycle_end:
                        logger.debug(
                            "[ws-decision] TERMINAL: chat_final + lifecycle:end seen, runId=%s",
                            run_id,
                        )
                        done = True
                    else:
                        logger.debug(
                            "[ws-decision] chat_final (step done), waiting for lifecycle:end, runId=%s",
                            run_id,
                        )
                else:
                    logger.debug(
                        "[ws-decision] chat event unhandled state=%s textLen=%d",
                        state, len(text) if text else 0,
                    )
                continue

            # --- Handle event:agent (lifecycle, tool, assistant) ---
            if ftype == "event" and event_name == "agent":
                payload = frame.get("payload", {})

                # ── Run-ID filtering (same logic as event:chat above) ──
                event_run_id = payload.get("runId", "")
                if not got_ack:
                    logger.debug(
                        "[ws-decision] SKIP pre-ACK agent event: runId=%s",
                        event_run_id,
                    )
                    continue
                if event_run_id and event_run_id != run_id:
                    logger.debug(
                        "[ws-decision] SKIP agent event wrong runId: got=%s want=%s",
                        event_run_id, run_id,
                    )
                    continue

                stream = payload.get("stream", "")
                data = payload.get("data", {})

                if stream == "lifecycle":
                    phase = data.get("phase", "")
                    logger.debug(
                        "[ws-decision] YIELD agent lifecycle phase=%s runId=%s",
                        phase, run_id,
                    )
                    if phase == "start":
                        event_key = ("lifecycle", run_id, phase)
                        budget = _replay_skip_budget.get(event_key, 0)
                        if _recv_retry_count > 0 and budget > 0:
                            _replay_skip_budget[event_key] = budget - 1
                            logger.debug(
                                "[ws-decision] SKIP replayed lifecycle event phase=%s runId=%s",
                                phase, run_id,
                            )
                            continue
                        _non_text_emitted_counts[event_key] = (
                            _non_text_emitted_counts.get(event_key, 0) + 1
                        )
                        yield OpenClawEvent(
                            type=OpenClawEventType.LIFECYCLE_START,
                            run_id=run_id,
                            raw=frame,
                        )
                    elif phase == "end":
                        event_key = ("lifecycle", run_id, phase)
                        budget = _replay_skip_budget.get(event_key, 0)
                        if _recv_retry_count > 0 and budget > 0:
                            _replay_skip_budget[event_key] = budget - 1
                            logger.debug(
                                "[ws-decision] SKIP replayed lifecycle event phase=%s runId=%s",
                                phase, run_id,
                            )
                            continue
                        _non_text_emitted_counts[event_key] = (
                            _non_text_emitted_counts.get(event_key, 0) + 1
                        )
                        yield OpenClawEvent(
                            type=OpenClawEventType.LIFECYCLE_END,
                            run_id=run_id,
                            raw=frame,
                        )
                        _got_lifecycle_end = True
                        if _got_any_chat_final:
                            logger.debug(
                                "[ws-decision] TERMINAL: lifecycle:end + chat_final seen, runId=%s",
                                run_id,
                            )
                            done = True

                elif stream == "tool":
                    logger.debug(
                        "[ws-decision] YIELD agent tool=%s runId=%s",
                        data.get("name", "?"), run_id,
                    )
                    tool_fingerprint = json.dumps(
                        {
                            "name": data.get("name", ""),
                            "args": data.get("args", {}),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    event_key = ("tool", run_id, tool_fingerprint)
                    budget = _replay_skip_budget.get(event_key, 0)
                    if _recv_retry_count > 0 and budget > 0:
                        _replay_skip_budget[event_key] = budget - 1
                        logger.debug(
                            "[ws-decision] SKIP replayed tool event name=%s runId=%s",
                            data.get("name", "?"), run_id,
                        )
                        continue
                    _non_text_emitted_counts[event_key] = (
                        _non_text_emitted_counts.get(event_key, 0) + 1
                    )
                    yield OpenClawEvent(
                        type=OpenClawEventType.TOOL_TRACE,
                        tool_name=data.get("name", ""),
                        tool_args=data.get("args", {}),
                        run_id=run_id,
                        raw=frame,
                    )

                elif stream == "assistant":
                    # Redundant with event:chat deltas — do not yield text here
                    # to avoid double-counting. event:chat is the canonical channel.
                    logger.debug("[ws-decision] SKIP agent assistant (redundant)")
                continue

            # --- Other events (health, presence, etc.) — skip ---
            logger.debug("[ws-decision] SKIP unrelated frame: type=%s event=%s", ftype, event_name)

    async def chat_inject(
        self,
        session_key: str,
        message: str,
        label: Optional[str] = None,
        timeout: float = 10.0,
    ) -> bool:
        """Inject a one-time assistant message into the session transcript.

        This enters model context on subsequent turns without running the agent.
        Useful for injecting session-level instructions (e.g. voice mode).

        Returns True if the inject was acknowledged.
        """
        await self.ensure_connected()

        req_id = self._next_req_id()
        params: dict = {
            "sessionKey": session_key,
            "message": message,
        }
        if label:
            params["label"] = label

        inject_req = {
            "type": "req",
            "id": req_id,
            "method": "chat.inject",
            "params": params,
        }

        assert self._ws is not None
        await self._ws.send(json.dumps(inject_req))
        logger.info("chat.inject sent: session=%s label=%s", session_key, label)

        deadline = asyncio.get_event_loop().time() + timeout
        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                assert self._ws is not None
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                res = json.loads(raw)
                if res.get("type") == "res" and res.get("id") == req_id:
                    if res.get("ok"):
                        logger.info("chat.inject acknowledged")
                        return True
                    logger.warning("chat.inject rejected: %s", res)
                    return False
                # Unrelated frame (health tick, stale event, etc.) — skip
                logger.debug("chat.inject: skipping unrelated frame: %s", res.get("type"))
        except asyncio.TimeoutError:
            pass
        logger.warning("chat.inject: no ACK for %s within %.0fs", req_id, timeout)
        return False

    async def chat_abort(self, run_id: str) -> bool:
        """
        Abort an in-flight chat run. Best-effort.

        Returns True if the abort request was acknowledged.
        """
        if not self.is_connected:
            return False

        req_id = self._next_req_id()
        abort_req = {
            "type": "req",
            "id": req_id,
            "method": "chat.abort",
            "params": {"runId": run_id},
        }

        try:
            assert self._ws is not None
            await self._ws.send(json.dumps(abort_req))
            logger.info("chat.abort sent for runId=%s", run_id)
            # Don't wait for response — best-effort
            return True
        except Exception as e:
            logger.warning("chat.abort failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_req_id(self) -> str:
        self._req_counter += 1
        return f"r{self._req_counter}"

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """Extract text from a chat event payload."""
        msg = payload.get("message", {})
        content_blocks = msg.get("content", [])
        parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_gateway_client_from_config(config: dict) -> OpenClawGatewayClient:
    """
    Create a gateway client from the scaffold config dict.

    Reads openclaw config from config.yaml's openclaw section,
    falling back to environment variables and ~/.openclaw/openclaw.json.
    """
    oc_cfg = config.get("openclaw", {})
    gateway_url = oc_cfg.get("gateway_ws_url", "ws://127.0.0.1:18789")

    # Token: env var > config.yaml > ~/.openclaw/openclaw.json
    token_env = oc_cfg.get("gateway_token_env", "OPENCLAW_GATEWAY_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        try:
            import pathlib
            oc_json = pathlib.Path.home() / ".openclaw" / "openclaw.json"
            if oc_json.exists():
                with open(oc_json) as f:
                    oc_data = json.load(f)
                token = oc_data.get("gateway", {}).get("auth", {}).get("token", "")
                if token:
                    logger.info("Loaded gateway token from %s", oc_json)
        except Exception as e:
            logger.debug("Could not load token from openclaw.json: %s", e)

    return OpenClawGatewayClient(
        gateway_url=gateway_url,
        token=token,
    )
