"""
Async WebSocket client for OpenClaw Gateway — Node role.

Registers Reachy as an OpenClaw node with camera.snap capability.
When the agent invokes camera_snap via the nodes tool, the Gateway
dispatches a node.invoke.request to this client, which captures a
frame from CameraWorker and returns a JPEG as base64.

Separate WS connection from the operator/backend client used for chat.
"""

import asyncio
import base64
import hashlib
import json
import logging
import pathlib
import time
from typing import Any, Optional

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ed25519 device identity (persistent keypair for unique node registration)
# ---------------------------------------------------------------------------

_IDENTITY_PATH = pathlib.Path.home() / ".openclaw" / "reachy-node-identity.json"


def _b64url_encode(data: bytes) -> str:
    """Base64-URL encode without padding (RFC 4648 §5)."""
    return base64.b64encode(data).decode("ascii").replace("+", "-").replace("/", "_").rstrip("=")


def _load_or_create_identity() -> dict:
    """Load or generate a persistent Ed25519 identity for this node.

    Returns dict with keys: device_id, public_key_b64url, private_key_pem, public_key_pem.
    """
    if _IDENTITY_PATH.exists():
        try:
            data = json.loads(_IDENTITY_PATH.read_text())
            if data.get("device_id") and data.get("private_key_pem"):
                logger.debug("Loaded node identity from %s (device_id=%s…)", _IDENTITY_PATH, data["device_id"][:12])
                return data
        except Exception as e:
            logger.warning("Failed to load node identity, regenerating: %s", e)

    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")

    # Raw 32-byte public key (strip SPKI DER prefix)
    public_der = public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    raw_pub = public_der[-32:]

    device_id = hashlib.sha256(raw_pub).hexdigest()
    pub_b64url = _b64url_encode(raw_pub)

    data = {
        "device_id": device_id,
        "public_key_b64url": pub_b64url,
        "private_key_pem": private_pem,
        "public_key_pem": public_pem,
    }

    _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _IDENTITY_PATH.write_text(json.dumps(data, indent=2))
    logger.info("Generated new node identity at %s (device_id=%s…)", _IDENTITY_PATH, device_id[:12])
    return data


def _sign_connect_payload(identity: dict, token: str, client_id: str, client_mode: str, role: str, scopes: list[str]) -> dict:
    """Build the device auth block for the connect handshake.

    Returns the ``device`` dict to include in ConnectParams.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization

    signed_at_ms = int(time.time() * 1000)
    scopes_str = ",".join(scopes)
    token_str = token or ""
    # v1 payload (no nonce — localhost)
    payload = f"v1|{identity['device_id']}|{client_id}|{client_mode}|{role}|{scopes_str}|{signed_at_ms}|{token_str}"

    private_key = serialization.load_pem_private_key(identity["private_key_pem"].encode(), password=None)
    assert isinstance(private_key, ed25519.Ed25519PrivateKey)
    signature = private_key.sign(payload.encode("utf-8"))

    return {
        "id": identity["device_id"],
        "publicKey": identity["public_key_b64url"],
        "signature": _b64url_encode(signature),
        "signedAt": signed_at_ms,
    }


class OpenClawNodeClient:
    """
    Persistent async WS client for OpenClaw Gateway in **node** role.

    Declares ``camera.snap``, ``camera.list``, ``action.perform``, and
    ``action.list`` as supported commands. When the Gateway dispatches a
    ``node.invoke.request``, the handler routes to the appropriate
    handler (camera capture or physical action).

    Usage::

        client = OpenClawNodeClient(gateway_url, token, camera_worker)
        await client.connect()
        asyncio.create_task(client.run())
        ...
        await client.stop()
    """

    CLIENT_ID = "node-host"  # Schema-enforced enum (required by Gateway)
    DISPLAY_NAME = "Reachy Mini Lite"

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18789",
        token: str = "",
        camera_worker: Any = None,
        movement_manager: Any = None,
        max_width: int = 768,
        jpeg_quality: int = 80,
    ):
        self.gateway_url = gateway_url
        self.token = token
        self.camera_worker = camera_worker
        self.movement_manager = movement_manager
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality

        self._identity = _load_or_create_identity()
        self.node_id = self._identity["device_id"]  # Unique, persistent

        self._ws: Optional[ClientConnection] = None
        self._connected = False
        self._req_counter = 0
        self._stopping = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Gateway and complete the protocol v3 handshake as a node."""
        if self._connected and self._ws:
            logger.debug("Node client already connected")
            return

        logger.info("Node client connecting to Gateway at %s ...", self.gateway_url)
        self._ws = await websockets.connect(self.gateway_url)

        # 1. Receive connect.challenge
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        challenge = json.loads(raw)
        if challenge.get("event") != "connect.challenge":
            raise ConnectionError(
                f"Expected connect.challenge, got: {challenge.get('event')}"
            )
        logger.debug("Node client got connect.challenge")

        # 2. Send connect request with node identity + device auth
        req_id = self._next_req_id()
        role = "node"
        scopes = ["operator.admin"]
        device_block = _sign_connect_payload(
            self._identity, self.token, self.CLIENT_ID, "node", role, scopes,
        )

        connect_req = {
            "type": "req",
            "id": req_id,
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": self.CLIENT_ID,
                    "displayName": self.DISPLAY_NAME,
                    "version": "0.1.0",
                    "platform": "linux",
                    "mode": "node",
                },
                "role": role,
                "scopes": scopes,
                "caps": ["camera", "action"],
                "commands": ["camera.snap", "camera.list", "action.perform", "action.list"],
                "permissions": {},
                "device": device_block,
                "auth": {"token": self.token} if self.token else {},
                "locale": "en-US",
                "userAgent": "reachy-node/0.1.0",
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
                f"Gateway rejected node connect: {error.get('message', error)}"
            )

        payload = res.get("payload", {})
        self._connected = True
        logger.info(
            "Node client connected (nodeId=%s…, displayName=%s, protocol=%s, server=%s)",
            self.node_id[:12],
            self.DISPLAY_NAME,
            payload.get("protocol"),
            payload.get("server", {}).get("version", "?"),
        )

        # 4. Drain initial events
        drained = 0
        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=0.3)
                drained += 1
            except asyncio.TimeoutError:
                break
        if drained:
            logger.debug("Node client drained %d initial events", drained)

    async def run(self) -> None:
        """
        Event loop: listen for Gateway messages and dispatch invoke requests.

        Run as ``asyncio.create_task(client.run())``.
        """
        if not self._connected or not self._ws:
            raise RuntimeError("Node client not connected — call connect() first")

        logger.info("Node client event loop started")
        try:
            while not self._stopping:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue  # keepalive; just loop

                frame = json.loads(raw)
                ftype = frame.get("type")
                event_name = frame.get("event", "")

                if ftype == "event" and event_name == "node.invoke.request":
                    payload = frame.get("payload", {})
                    asyncio.create_task(self._handle_invoke(payload))
                else:
                    logger.debug(
                        "Node client ignoring frame: type=%s event=%s",
                        ftype,
                        event_name,
                    )
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("Node client WS closed: %s", e)
        except asyncio.CancelledError:
            logger.info("Node client event loop cancelled")
            raise
        finally:
            self._connected = False
            logger.info("Node client event loop exited")

    async def stop(self) -> None:
        """Gracefully stop the event loop and close the WS."""
        self._stopping = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False
        logger.info("Node client stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    # ------------------------------------------------------------------
    # Invoke dispatch
    # ------------------------------------------------------------------

    async def _handle_invoke(self, payload: dict) -> None:
        """Dispatch a node.invoke.request to the appropriate handler."""
        req_id = payload.get("id", "")
        node_id = payload.get("nodeId", self.node_id)
        command = payload.get("command", "")
        params_json = payload.get("paramsJSON")
        params = json.loads(params_json) if params_json else {}

        logger.info("Node invoke: command=%s id=%s params=%s", command, req_id, params)

        try:
            if command == "camera.snap":
                result = await self._handle_camera_snap(params)
            elif command == "camera.list":
                result = self._handle_camera_list()
            elif command == "action.perform":
                result = self._handle_action_perform(params)
            elif command == "action.list":
                result = self._handle_action_list()
            else:
                result = {
                    "ok": False,
                    "error": {"code": "UNKNOWN_COMMAND", "message": f"Unknown command: {command}"},
                }

            await self._send_invoke_result(req_id, node_id, result)

        except Exception as e:
            logger.error("Node invoke error (command=%s): %s", command, e, exc_info=True)
            await self._send_invoke_result(req_id, node_id, {
                "ok": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(e)},
            })

    async def _handle_camera_snap(self, params: dict) -> dict:
        """Capture a frame from CameraWorker and return as base64 JPEG."""
        if self.camera_worker is None:
            return {
                "ok": False,
                "error": {"code": "NO_CAMERA", "message": "Camera worker not available"},
            }

        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            return {
                "ok": False,
                "error": {"code": "NO_FRAME", "message": "No frame available from camera"},
            }

        # Respect params from the invoke request (match mobile node schema)
        max_width = params.get("maxWidth", self.max_width)
        quality = params.get("quality")
        if quality is not None:
            # Mobile sends 0.0-1.0 float, convert to 1-100 int
            jpeg_quality = max(1, min(100, int(float(quality) * 100)))
        else:
            jpeg_quality = self.jpeg_quality

        from sparky_mvp.robot.scene_capture import encode_frame_as_image_block

        image_block = encode_frame_as_image_block(
            frame, max_size=max_width, jpeg_quality=jpeg_quality,
        )

        # Extract base64 data from Anthropic image block format
        b64_data = image_block["source"]["data"]

        # Compute actual dimensions after resize
        import cv2
        h, w = frame.shape[:2]
        if max(h, w) > max_width:
            scale = max_width / max(h, w)
            w = int(w * scale)
            h = int(h * scale)

        payload_dict = {
            "base64": b64_data,
            "width": w,
            "height": h,
            "format": "jpg",
        }

        return {
            "ok": True,
            "payloadJSON": json.dumps(payload_dict),
        }

    def _handle_camera_list(self) -> dict:
        """Return the list of available cameras (single front-facing camera)."""
        cameras = [{"id": "front", "facing": "front", "label": "Reachy head camera"}]
        return {"ok": True, "payloadJSON": json.dumps({"cameras": cameras})}

    # ------------------------------------------------------------------
    # Action handlers (physical movements)
    # ------------------------------------------------------------------

    def _handle_action_perform(self, params: dict) -> dict:
        """Perform a gentle physical action on the robot."""
        if self.movement_manager is None:
            return {
                "ok": False,
                "error": {"code": "NO_MOVEMENT_MANAGER", "message": "Movement manager not available"},
            }

        action_name = params.get("action", "")
        if not action_name:
            return {
                "ok": False,
                "error": {"code": "MISSING_ACTION", "message": "action parameter is required"},
            }

        from sparky_mvp.robot.gentle_actions import GENTLE_ACTIONS

        action_def = GENTLE_ACTIONS.get(action_name)
        if action_def is None:
            available = list(GENTLE_ACTIONS.keys())
            return {
                "ok": False,
                "error": {
                    "code": "UNKNOWN_ACTION",
                    "message": f"Unknown action: {action_name}. Available: {available}",
                },
            }

        try:
            # Support both nested {"action":"nod","params":{"nods":3}} and
            # flat {"action":"nod","nods":3} param styles
            action_params = params.get("params", {})
            if not action_params:
                action_params = {k: v for k, v in params.items() if k != "action"}
            move = action_def["create"](action_params)
            self.movement_manager.queue_move(move)
            logger.info("Action queued: %s (duration=%.1fs)", action_name, move.duration)
            return {
                "ok": True,
                "payloadJSON": json.dumps({
                    "action": action_name,
                    "status": "queued",
                    "duration": move.duration,
                }),
            }
        except Exception as e:
            logger.error("Action perform error: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": {"code": "ACTION_ERROR", "message": str(e)},
            }

    def _handle_action_list(self) -> dict:
        """Return the list of available actions."""
        from sparky_mvp.robot.gentle_actions import GENTLE_ACTIONS

        actions = []
        for name, defn in GENTLE_ACTIONS.items():
            actions.append({
                "name": name,
                "description": defn["description"],
            })
        return {"ok": True, "payloadJSON": json.dumps({"actions": actions})}

    # ------------------------------------------------------------------
    # Result sending
    # ------------------------------------------------------------------

    async def _send_invoke_result(self, req_id: str, node_id: str, result: dict) -> None:
        """Send a node.invoke.result RPC back to the Gateway."""
        if not self._ws:
            logger.warning("Cannot send invoke result — WS not connected")
            return

        params: dict = {
            "id": req_id,
            "nodeId": node_id,
            "ok": result.get("ok", False),
        }
        if "payloadJSON" in result:
            params["payloadJSON"] = result["payloadJSON"]
        if "error" in result:
            params["error"] = result["error"]

        result_req = {
            "type": "req",
            "id": self._next_req_id(),
            "method": "node.invoke.result",
            "params": params,
        }

        await self._ws.send(json.dumps(result_req))
        logger.info(
            "Node invoke result sent: id=%s ok=%s",
            req_id,
            result.get("ok"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_req_id(self) -> str:
        self._req_counter += 1
        return f"n{self._req_counter}"
