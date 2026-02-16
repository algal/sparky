#!/usr/bin/env python3
"""
Smoke test: register Reachy as an OpenClaw node with camera.snap capability.

Usage:
    cd forks/reachy-glados-example
    PYTHONPATH=. .venv/bin/python tools/node_camera_smoke_test.py

    # With real robot camera:
    PYTHONPATH=. .venv/bin/python tools/node_camera_smoke_test.py --robot

    # Then trigger via CLI in another terminal:
    openclaw nodes invoke --node node-host --command camera.snap --params '{}'

Environment:
    OPENCLAW_GATEWAY_TOKEN  -- optional (auto-loaded from ~/.openclaw/openclaw.json)
    OPENCLAW_GATEWAY_URL    -- optional (default: ws://127.0.0.1:18789)

Exit codes:
    0 = success (node registered, event loop running, optionally handled a snap)
    1 = failure
"""

import argparse
import asyncio
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("node_smoke")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sparky_mvp.core.openclaw_node_client import OpenClawNodeClient


def _load_token() -> str:
    """Load gateway token from env or ~/.openclaw/openclaw.json."""
    token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip()
    if token:
        return token
    try:
        import pathlib
        oc_json = pathlib.Path.home() / ".openclaw" / "openclaw.json"
        if oc_json.exists():
            with open(oc_json) as f:
                data = json.load(f)
            token = data.get("gateway", {}).get("auth", {}).get("token", "")
            if token:
                log.info("Loaded token from %s", oc_json)
                return token
    except Exception as e:
        log.debug("Could not load token: %s", e)
    return ""


class FakeCameraWorker:
    """Returns a synthetic test frame (colored gradient) for testing without robot."""

    def get_latest_frame(self):
        import numpy as np
        # 480x640 BGR gradient image
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Blue gradient left to right
        frame[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        # Green gradient top to bottom
        frame[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8), (w, 1)).T
        # Red constant
        frame[:, :, 2] = 128
        return frame


async def smoke_test(use_robot: bool = False):
    gateway_url = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789")
    token = _load_token()

    # Set up camera worker
    if use_robot:
        log.info("Using real robot camera")
        try:
            from reachy_mini import ReachyMini
            mini = ReachyMini(media_backend="default")
            from sparky_mvp.robot.camera_worker import CameraWorker
            camera_worker = CameraWorker(reachy_mini=mini, head_tracker=None)
            camera_worker.start()
            log.info("CameraWorker started")
            # Wait for first frame
            await asyncio.sleep(1.0)
        except Exception as e:
            log.error("Failed to start robot camera: %s", e)
            return False
    else:
        log.info("Using fake camera (synthetic gradient image)")
        camera_worker = FakeCameraWorker()

    # Create and connect node client
    client = OpenClawNodeClient(
        gateway_url=gateway_url,
        token=token,
        camera_worker=camera_worker,
    )

    try:
        log.info("Connecting to Gateway as node ...")
        await client.connect()
        log.info("Node registered successfully!")
        log.info("")
        log.info("Node is listening for invoke requests.")
        log.info("Trigger a camera.snap from another terminal:")
        log.info("  openclaw nodes invoke --node node-host --command camera.snap")
        log.info("")
        log.info("Press Ctrl+C to stop.")

        # Run event loop (blocks until stopped)
        await client.run()

    except KeyboardInterrupt:
        log.info("Interrupted")
    except ConnectionError as e:
        log.error("Connection failed: %s", e)
        return False
    finally:
        await client.stop()
        if use_robot and hasattr(camera_worker, 'stop'):
            camera_worker.stop()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node camera smoke test")
    parser.add_argument("--robot", action="store_true", help="Use real robot camera")
    args = parser.parse_args()

    try:
        ok = asyncio.run(smoke_test(use_robot=args.robot))
    except KeyboardInterrupt:
        ok = True
    sys.exit(0 if ok else 1)
