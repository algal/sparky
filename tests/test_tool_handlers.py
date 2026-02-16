"""
Unit tests for tool handlers module (simplified to datetime only).
"""

import pytest
from datetime import datetime

from sparky_mvp.tools.tool_handlers import (
    get_time,
    get_date,
    get_tool_handler,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS
)


class TestToolHandlers:
    """Tests for individual tool handler functions."""

    @pytest.mark.asyncio
    async def test_get_time(self):
        """Test get_time returns current time."""
        result = await get_time()

        assert isinstance(result, str)
        assert "The current time is" in result
        # Should contain time format like "03:45 PM"
        assert "M" in result  # AM or PM

    @pytest.mark.asyncio
    async def test_get_date(self):
        """Test get_date returns current date."""
        result = await get_date()

        assert isinstance(result, str)
        assert "Today is" in result
        # Should contain day of week and year
        assert str(datetime.now().year) in result


class TestToolRegistry:
    """Tests for tool registry and definitions."""

    def test_tool_handlers_registry_exists(self):
        """Test TOOL_HANDLERS registry is properly defined."""
        assert isinstance(TOOL_HANDLERS, dict)
        assert len(TOOL_HANDLERS) >= 2

        # Verify expected tools are registered
        expected_tools = ["get_time", "get_date", "look_at_scene"]

        for tool_name in expected_tools:
            assert tool_name in TOOL_HANDLERS
            assert callable(TOOL_HANDLERS[tool_name])

    def test_tool_definitions_format(self):
        """Test TOOL_DEFINITIONS has correct Anthropic format."""
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) >= 2

        for tool_def in TOOL_DEFINITIONS:
            # Verify structure
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def

            # Verify input schema
            schema = tool_def["input_schema"]
            assert isinstance(schema, dict)
            assert schema.get("type") == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_tool_definitions_match_handlers(self):
        """Test all tool definitions have matching handlers."""
        for tool_def in TOOL_DEFINITIONS:
            tool_name = tool_def["name"]
            assert tool_name in TOOL_HANDLERS, f"Missing handler for {tool_name}"

    def test_all_handlers_have_definitions(self):
        """Test all handlers have corresponding definitions."""
        defined_tools = {tool_def["name"] for tool_def in TOOL_DEFINITIONS}

        for handler_name in TOOL_HANDLERS.keys():
            assert handler_name in defined_tools, f"Missing definition for {handler_name}"

    def test_get_tool_handler_exists(self):
        """Test get_tool_handler retrieves existing tools."""
        handler = get_tool_handler("get_time")

        assert handler is not None
        assert callable(handler)
        assert handler == get_time

    def test_get_tool_handler_not_exists(self):
        """Test get_tool_handler returns None for non-existent tools."""
        handler = get_tool_handler("nonexistent_tool")

        assert handler is None

    @pytest.mark.asyncio
    async def test_all_handlers_are_async(self):
        """Test all tool handlers are async functions."""
        import inspect

        for tool_name, handler in TOOL_HANDLERS.items():
            assert inspect.iscoroutinefunction(handler), \
                f"{tool_name} handler is not async"

    def test_tool_definitions_have_descriptions(self):
        """Test all tools have meaningful descriptions."""
        for tool_def in TOOL_DEFINITIONS:
            desc = tool_def["description"]
            assert isinstance(desc, str)
            assert len(desc) > 5  # Meaningful description


class TestToolIntegration:
    """Integration tests for tool handlers."""

    @pytest.mark.asyncio
    async def test_all_handlers_execute_without_error(self):
        """Test all handlers can be called and return strings."""
        # Test handlers with minimal valid arguments
        test_cases = [
            ("get_time", {}),
            ("get_date", {}),
        ]

        for tool_name, kwargs in test_cases:
            handler = get_tool_handler(tool_name)
            assert handler is not None, f"Handler not found: {tool_name}"

            result = await handler(**kwargs)
            assert isinstance(result, str), \
                f"{tool_name} did not return string: {type(result)}"
            assert len(result) > 0, f"{tool_name} returned empty string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
