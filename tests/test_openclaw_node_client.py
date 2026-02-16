"""
Unit tests for OpenClaw Node Client.

Tests the node handshake, invoke dispatch, camera.snap handler,
error handling, and result sending. Uses a mock WebSocket.
"""

import asyncio
import json
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from sparky_mvp.core.openclaw_node_client import OpenClawNodeClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_challenge() -> str:
    return json.dumps({"event": "connect.challenge", "payload": {"nonce": "abc"}})


def _connect_response(req_id: str, ok: bool = True) -> str:
    frame = {"type": "res", "id": req_id, "ok": ok}
    if ok:
        frame["payload"] = {"protocol": 3, "server": {"version": "1.0.0"}}
    else:
        frame["error"] = {"message": "auth failed"}
    return json.dumps(frame)


def _invoke_request(
    req_id: str = "invoke-1",
    command: str = "camera.snap",
    params: dict | None = None,
    node_id: str = "node-host",
) -> str:
    payload = {
        "id": req_id,
        "nodeId": node_id,
        "command": command,
        "timeoutMs": 30000,
    }
    if params is not None:
        payload["paramsJSON"] = json.dumps(params)
    return json.dumps({
        "type": "event",
        "event": "node.invoke.request",
        "payload": payload,
    })


class MockWebSocket:
    """Simulates a WebSocket connection with a sequence of frames."""

    def __init__(self, frames: list[str]):
        self._frames = list(frames)
        self._idx = 0
        self.sent: list[str] = []
        self._closed = False

    async def recv(self) -> str:
        if self._closed:
            raise Exception("Connection closed")
        if self._idx >= len(self._frames):
            await asyncio.sleep(999)
        frame = self._frames[self._idx]
        self._idx += 1
        if frame == "__TIMEOUT__":
            await asyncio.sleep(999)
        return frame

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self._closed = True


class FakeCameraWorker:
    """Returns a synthetic test frame."""

    def __init__(self, return_frame: bool = True):
        self._return_frame = return_frame

    def get_latest_frame(self):
        if not self._return_frame:
            return None
        return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Handshake tests
# ---------------------------------------------------------------------------


class TestNodeConnect:

    @pytest.mark.asyncio
    async def test_successful_handshake(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("n1"),
        ])
        with patch("sparky_mvp.core.openclaw_node_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawNodeClient(token="secret", camera_worker=FakeCameraWorker())
            await client.connect()

            assert client.is_connected
            sent = json.loads(ws.sent[0])
            assert sent["method"] == "connect"
            assert sent["params"]["client"]["id"] == "node-host"
            assert sent["params"]["client"]["mode"] == "node"
            assert sent["params"]["client"]["displayName"] == "Reachy Mini Lite"
            assert sent["params"]["role"] == "node"
            assert sent["params"]["caps"] == ["camera", "action"]
            assert sent["params"]["commands"] == ["camera.snap", "camera.list", "action.perform", "action.list"]
            assert sent["params"]["auth"]["token"] == "secret"
            # Device identity block
            assert "device" in sent["params"]
            device = sent["params"]["device"]
            assert device["id"] == client.node_id
            assert "publicKey" in device
            assert "signature" in device
            assert "signedAt" in device

    @pytest.mark.asyncio
    async def test_bad_challenge_raises(self):
        ws = MockWebSocket([
            json.dumps({"event": "not.a.challenge"}),
        ])
        with patch("sparky_mvp.core.openclaw_node_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawNodeClient()
            with pytest.raises(ConnectionError, match="Expected connect.challenge"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_rejected_connect_raises(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("n1", ok=False),
        ])
        with patch("sparky_mvp.core.openclaw_node_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawNodeClient()
            with pytest.raises(ConnectionError, match="auth failed"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_idempotent_connect(self):
        ws = MockWebSocket([
            _connect_challenge(),
            _connect_response("n1"),
        ])
        with patch("sparky_mvp.core.openclaw_node_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock(return_value=ws)
            client = OpenClawNodeClient()
            await client.connect()
            await client.connect()
            assert mock_ws.connect.call_count == 1


# ---------------------------------------------------------------------------
# Invoke dispatch tests
# ---------------------------------------------------------------------------


class TestInvokeDispatch:

    @pytest.mark.asyncio
    async def test_camera_snap_returns_base64(self):
        """camera.snap should return ok=true with base64 JPEG payload."""
        camera = FakeCameraWorker()
        client = OpenClawNodeClient(camera_worker=camera)
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "camera.snap",
        })

        # Should have sent one result
        assert len(client._ws.sent) == 1
        result_req = json.loads(client._ws.sent[0])
        assert result_req["method"] == "node.invoke.result"
        params = result_req["params"]
        assert params["id"] == "req-1"
        assert params["nodeId"] == "node-host"
        assert params["ok"] is True

        # Verify payload JSON
        payload = json.loads(params["payloadJSON"])
        assert "base64" in payload
        assert payload["format"] == "jpg"
        assert isinstance(payload["width"], int)
        assert isinstance(payload["height"], int)
        assert len(payload["base64"]) > 0

    @pytest.mark.asyncio
    async def test_camera_snap_respects_max_width(self):
        """camera.snap should respect maxWidth param."""
        camera = FakeCameraWorker()
        client = OpenClawNodeClient(camera_worker=camera)
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "camera.snap",
            "paramsJSON": json.dumps({"maxWidth": 320}),
        })

        result_req = json.loads(client._ws.sent[0])
        payload = json.loads(result_req["params"]["payloadJSON"])
        assert payload["width"] <= 320
        assert payload["height"] <= 320

    @pytest.mark.asyncio
    async def test_camera_snap_respects_quality(self):
        """camera.snap should respect quality param (0.0-1.0 float)."""
        camera = FakeCameraWorker()
        client = OpenClawNodeClient(camera_worker=camera)
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "camera.snap",
            "paramsJSON": json.dumps({"quality": 0.5}),
        })

        # Should succeed (quality applied internally)
        result_req = json.loads(client._ws.sent[0])
        assert result_req["params"]["ok"] is True

    @pytest.mark.asyncio
    async def test_no_camera_worker_returns_error(self):
        """camera.snap without camera_worker should return NO_CAMERA error."""
        client = OpenClawNodeClient(camera_worker=None)
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "camera.snap",
        })

        result_req = json.loads(client._ws.sent[0])
        params = result_req["params"]
        assert params["ok"] is False
        assert params["error"]["code"] == "NO_CAMERA"

    @pytest.mark.asyncio
    async def test_no_frame_returns_error(self):
        """camera.snap with no frame available should return NO_FRAME error."""
        camera = FakeCameraWorker(return_frame=False)
        client = OpenClawNodeClient(camera_worker=camera)
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "camera.snap",
        })

        result_req = json.loads(client._ws.sent[0])
        params = result_req["params"]
        assert params["ok"] is False
        assert params["error"]["code"] == "NO_FRAME"

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self):
        """Unknown commands should return UNKNOWN_COMMAND error."""
        client = OpenClawNodeClient(camera_worker=FakeCameraWorker())
        client._ws = MockWebSocket([])
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "system.run",
        })

        result_req = json.loads(client._ws.sent[0])
        params = result_req["params"]
        assert params["ok"] is False
        assert params["error"]["code"] == "UNKNOWN_COMMAND"


# ---------------------------------------------------------------------------
# Event loop tests
# ---------------------------------------------------------------------------


class TestEventLoop:

    @pytest.mark.asyncio
    async def test_run_dispatches_invoke_requests(self):
        """run() should dispatch node.invoke.request events."""
        camera = FakeCameraWorker()
        invoke_frame = _invoke_request("req-abc", "camera.snap")

        ws = MockWebSocket([invoke_frame])
        client = OpenClawNodeClient(camera_worker=camera)
        client._ws = ws
        client._connected = True

        # Run for a short time then stop
        async def stop_after_delay():
            await asyncio.sleep(0.2)
            client._stopping = True

        asyncio.create_task(stop_after_delay())

        await client.run()

        # Should have sent a result
        assert len(ws.sent) >= 1
        result = json.loads(ws.sent[0])
        assert result["method"] == "node.invoke.result"
        assert result["params"]["id"] == "req-abc"
        assert result["params"]["ok"] is True

    @pytest.mark.asyncio
    async def test_run_ignores_non_invoke_events(self):
        """run() should skip non-invoke events."""
        ws = MockWebSocket([
            json.dumps({"type": "event", "event": "health", "payload": {}}),
        ])
        client = OpenClawNodeClient()
        client._ws = ws
        client._connected = True

        async def stop_after_delay():
            await asyncio.sleep(0.2)
            client._stopping = True

        asyncio.create_task(stop_after_delay())
        await client.run()

        # No results sent
        assert len(ws.sent) == 0


# ---------------------------------------------------------------------------
# Stop / cleanup
# ---------------------------------------------------------------------------


class TestStopCleanup:

    @pytest.mark.asyncio
    async def test_stop_closes_ws(self):
        ws = MockWebSocket([])
        client = OpenClawNodeClient()
        client._ws = ws
        client._connected = True

        await client.stop()
        assert not client.is_connected
        assert ws._closed
