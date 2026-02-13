"""Tool registry for the agentic trading loop.

Provides a registry that maps tool names to handlers and generates
Anthropic-compatible tool definitions for the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """A single tool available to the agent."""

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]


class ToolRegistry:
    """Registry of tools available to the agent loop.

    Tools are registered with a name, description, JSON Schema for inputs,
    and an async handler function.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name (must be unique).
            description: Human-readable description for the model.
            input_schema: JSON Schema dict for tool inputs.
            handler: Async callable that takes **kwargs from input_schema.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    def get_tool_definitions(self) -> list[dict]:
        """Return tool definitions in Anthropic API format.

        Returns:
            List of dicts with name, description, input_schema.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> Any:
        """Execute a tool by name with the given arguments.

        Args:
            name: Tool name.
            arguments: Dict of arguments matching the tool's input_schema.

        Returns:
            Tool result (JSON-serializable).

        Raises:
            KeyError: If tool name is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        tool = self._tools[name]
        return await tool.handler(**arguments)

    @property
    def tool_names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())
