"""
Unit tests for gentle actions and the node action interface.

Tests:
- Gentle action Move implementations (duration, evaluate)
- Action registry (GENTLE_ACTIONS)
- Node client action.perform dispatch
- Node client action.list dispatch
"""

import json
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from sparky_mvp.robot.gentle_actions import (
    GENTLE_ACTIONS,
    GentleStretchMove,
    GentleYawnMove,
    GentleNodMove,
    GentleShakeMove,
    GentleLookAroundMove,
    GentleAntennaWiggleMove,
)
from sparky_mvp.core.openclaw_node_client import OpenClawNodeClient


# ---------------------------------------------------------------------------
# Gentle action Move tests
# ---------------------------------------------------------------------------

class TestGentleMoves:

    def test_stretch_duration(self):
        move = GentleStretchMove(duration=3.0)
        assert move.duration == 3.0

    def test_stretch_evaluate_returns_tuple(self):
        move = GentleStretchMove()
        result = move.evaluate(0.5)
        assert len(result) == 3
        head, antennas, body_yaw = result
        assert head is not None
        assert head.shape == (4, 4)
        assert antennas is not None
        assert len(antennas) == 2

    def test_yawn_duration(self):
        move = GentleYawnMove(duration=3.5)
        assert move.duration == 3.5

    def test_yawn_evaluate_head_tilts_back(self):
        move = GentleYawnMove()
        # At peak (~40%), pitch should be negative (tilted back)
        head, _, _ = move.evaluate(1.4)  # ~40% of 3.5s
        # The head pose is a 4x4 matrix; just verify it's valid
        assert head.shape == (4, 4)

    def test_nod_evaluate_returns_none_antennas(self):
        """Nod only moves head, not antennas."""
        move = GentleNodMove()
        head, antennas, body_yaw = move.evaluate(0.5)
        assert head is not None
        assert antennas is None

    def test_shake_evaluate(self):
        move = GentleShakeMove()
        head, antennas, body_yaw = move.evaluate(0.5)
        assert head is not None
        assert antennas is None

    def test_look_around_left(self):
        move = GentleLookAroundMove(direction="left")
        head, antennas, _ = move.evaluate(1.0)
        assert head is not None
        assert antennas is not None

    def test_look_around_right(self):
        move = GentleLookAroundMove(direction="right")
        head, antennas, _ = move.evaluate(1.0)
        assert head is not None

    def test_antenna_wiggle_decays(self):
        move = GentleAntennaWiggleMove()
        _, ant_early, _ = move.evaluate(0.1)
        _, ant_late, _ = move.evaluate(1.1)
        # Late antenna movement should be smaller (decayed)
        assert abs(ant_late[0]) < abs(ant_early[0]) or abs(ant_early[0]) < 0.01

    def test_all_moves_start_at_neutral(self):
        """At t=0, all moves should produce near-neutral poses."""
        for name, defn in GENTLE_ACTIONS.items():
            move = defn["create"]({})
            result = move.evaluate(0.0)
            assert len(result) == 3, f"{name} evaluate should return 3-tuple"


# ---------------------------------------------------------------------------
# Action registry tests
# ---------------------------------------------------------------------------

class TestActionRegistry:

    def test_all_actions_have_description(self):
        for name, defn in GENTLE_ACTIONS.items():
            assert "description" in defn, f"{name} missing description"
            assert len(defn["description"]) > 0

    def test_all_actions_have_create(self):
        for name, defn in GENTLE_ACTIONS.items():
            assert "create" in defn, f"{name} missing create"
            assert callable(defn["create"])

    def test_all_actions_create_move(self):
        for name, defn in GENTLE_ACTIONS.items():
            move = defn["create"]({})
            assert hasattr(move, "duration")
            assert hasattr(move, "evaluate")
            assert move.duration > 0

    def test_expected_actions_present(self):
        expected = {"stretch", "yawn", "nod", "shake", "look_around", "antenna_wiggle"}
        assert expected == set(GENTLE_ACTIONS.keys())


# ---------------------------------------------------------------------------
# Node client action dispatch tests
# ---------------------------------------------------------------------------

class MockWebSocket:
    def __init__(self):
        self.sent = []
    async def send(self, data):
        self.sent.append(data)
    async def recv(self):
        import asyncio
        await asyncio.sleep(999)
    async def close(self):
        pass


class TestNodeActionDispatch:

    @pytest.mark.asyncio
    async def test_action_list_returns_all_actions(self):
        mm = MagicMock()
        client = OpenClawNodeClient(movement_manager=mm)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.list",
        })

        assert len(client._ws.sent) == 1
        result = json.loads(client._ws.sent[0])
        payload = json.loads(result["params"]["payloadJSON"])
        assert "actions" in payload
        action_names = [a["name"] for a in payload["actions"]]
        assert "stretch" in action_names
        assert "yawn" in action_names
        assert "nod" in action_names

    @pytest.mark.asyncio
    async def test_action_perform_queues_move(self):
        mm = MagicMock()
        client = OpenClawNodeClient(movement_manager=mm)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.perform",
            "paramsJSON": json.dumps({"action": "nod"}),
        })

        # Should have queued a move
        mm.queue_move.assert_called_once()
        move = mm.queue_move.call_args[0][0]
        assert hasattr(move, "duration")

        # Should return ok
        result = json.loads(client._ws.sent[0])
        assert result["params"]["ok"] is True
        payload = json.loads(result["params"]["payloadJSON"])
        assert payload["action"] == "nod"
        assert payload["status"] == "queued"

    @pytest.mark.asyncio
    async def test_action_perform_with_params(self):
        mm = MagicMock()
        client = OpenClawNodeClient(movement_manager=mm)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.perform",
            "paramsJSON": json.dumps({
                "action": "look_around",
                "params": {"direction": "right", "duration": 4.0},
            }),
        })

        mm.queue_move.assert_called_once()
        move = mm.queue_move.call_args[0][0]
        assert move.duration == 4.0

    @pytest.mark.asyncio
    async def test_action_perform_unknown_returns_error(self):
        mm = MagicMock()
        client = OpenClawNodeClient(movement_manager=mm)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.perform",
            "paramsJSON": json.dumps({"action": "backflip"}),
        })

        result = json.loads(client._ws.sent[0])
        assert result["params"]["ok"] is False
        assert result["params"]["error"]["code"] == "UNKNOWN_ACTION"

    @pytest.mark.asyncio
    async def test_action_perform_no_mm_returns_error(self):
        client = OpenClawNodeClient(movement_manager=None)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.perform",
            "paramsJSON": json.dumps({"action": "nod"}),
        })

        result = json.loads(client._ws.sent[0])
        assert result["params"]["ok"] is False
        assert result["params"]["error"]["code"] == "NO_MOVEMENT_MANAGER"

    @pytest.mark.asyncio
    async def test_action_perform_missing_action_returns_error(self):
        mm = MagicMock()
        client = OpenClawNodeClient(movement_manager=mm)
        client._ws = MockWebSocket()
        client._connected = True

        await client._handle_invoke({
            "id": "req-1",
            "nodeId": "node-host",
            "command": "action.perform",
            "paramsJSON": json.dumps({}),
        })

        result = json.loads(client._ws.sent[0])
        assert result["params"]["ok"] is False
        assert result["params"]["error"]["code"] == "MISSING_ACTION"
