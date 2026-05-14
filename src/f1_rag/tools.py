"""Tool definitions for the v2 agentic pipeline.

Each tool wraps one ``JolpicaClient`` method. Definitions follow the
OpenAI tool-calling schema (which Groq and OpenAI both consume); the LLM
reads ``description`` to decide when to call. Keep descriptions tight --
verbose ones distract the model.

``TOOL_REGISTRY`` maps tool name -> Python callable that takes the parsed
arguments dict and returns a JSON-serialisable dict (which gets stringified
and fed back to the LLM as a ``tool`` message).
"""

from __future__ import annotations

from typing import Any, Callable

from f1_rag.jolpica import JolpicaClient

# ---------- OpenAI-format tool definitions ----------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_race_results",
            "description": (
                "Get the final finishing order of one specific F1 race, including each "
                "driver's position, points, grid (starting) position, time/gap, status, "
                "and constructor. Use for factual questions about race winners, podiums, "
                "or specific finishing positions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "integer",
                        "description": "F1 season year, e.g. 2022.",
                    },
                    "round": {
                        "type": "integer",
                        "description": "Round number within the season (1 = season opener).",
                    },
                },
                "required": ["season", "round"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_qualifying",
            "description": (
                "Get the qualifying-session results for one specific F1 race: each driver's "
                "qualifying position and Q1/Q2/Q3 lap times. Use for questions about pole "
                "position, qualifying order, or specific qualifying performances."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "F1 season year."},
                    "round": {"type": "integer", "description": "Round number within the season."},
                },
                "required": ["season", "round"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_driver_standings",
            "description": (
                "Get the final (or current-as-of-latest-race) Drivers' Championship standings "
                "for a season: each driver's position, points, and wins. Use for questions "
                "about who won a season, championship margins, or final season position."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "F1 season year."},
                },
                "required": ["season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_constructor_standings",
            "description": (
                "Get the final (or current) Constructors' Championship standings for a season: "
                "each team's position, points, and wins. Use for questions about constructor "
                "championships or team season performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "F1 season year."},
                },
                "required": ["season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_schedule",
            "description": (
                "Get the round-by-round calendar for a season: race name, circuit, location, "
                "and date for each round. Use for questions about a season's calendar, race "
                "ordering, or what races took place where."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "F1 season year."},
                },
                "required": ["season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_driver_info",
            "description": (
                "Get biographical info for one driver: full name, code, nationality, date of "
                "birth. The driver_id is a slug like 'max_verstappen' or 'lewis_hamilton'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "driver_id": {
                        "type": "string",
                        "description": (
                            "Driver ID slug (lowercase, underscore-separated), "
                            "e.g. 'max_verstappen', 'lewis_hamilton', 'charles_leclerc'."
                        ),
                    },
                },
                "required": ["driver_id"],
            },
        },
    },
]


# ---------- executor registry ----------

ToolFn = Callable[[JolpicaClient, dict[str, Any]], dict[str, Any]]


def _exec_race_results(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    results = client.get_race_results(season=args["season"], round_=args["round"])
    return {
        "season": args["season"],
        "round": args["round"],
        "n_results": len(results),
        "results": [r.model_dump() for r in results],
    }


def _exec_qualifying(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    results = client.get_qualifying(season=args["season"], round_=args["round"])
    return {
        "season": args["season"],
        "round": args["round"],
        "n_results": len(results),
        "qualifying": [q.model_dump() for q in results],
    }


def _exec_driver_standings(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    standings = client.get_driver_standings(season=args["season"])
    return {"season": args["season"], "standings": [s.model_dump() for s in standings]}


def _exec_constructor_standings(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    standings = client.get_constructor_standings(season=args["season"])
    return {"season": args["season"], "standings": [s.model_dump() for s in standings]}


def _exec_race_schedule(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    schedule = client.get_race_schedule(season=args["season"])
    return {"season": args["season"], "schedule": [r.model_dump() for r in schedule]}


def _exec_driver_info(client: JolpicaClient, args: dict[str, Any]) -> dict[str, Any]:
    info = client.get_driver_info(driver_id=args["driver_id"])
    if info is None:
        return {"driver_id": args["driver_id"], "found": False}
    return {"driver_id": args["driver_id"], "found": True, "driver": info.model_dump()}


TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_race_results": _exec_race_results,
    "get_qualifying": _exec_qualifying,
    "get_driver_standings": _exec_driver_standings,
    "get_constructor_standings": _exec_constructor_standings,
    "get_race_schedule": _exec_race_schedule,
    "get_driver_info": _exec_driver_info,
}


def execute_tool(name: str, arguments: dict[str, Any], client: JolpicaClient) -> dict[str, Any]:
    """Look up the tool by name and run it. Raises ``KeyError`` on unknown tools."""
    if name not in TOOL_REGISTRY:
        raise KeyError(f"unknown tool: {name!r}")
    return TOOL_REGISTRY[name](client, arguments)
