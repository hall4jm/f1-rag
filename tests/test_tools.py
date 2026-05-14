"""Unit tests for tools.py -- the OpenAI tool definitions and the executor registry.

Each test stubs the JolpicaClient so we don't make real HTTP calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from f1_rag.jolpica import (
    ConstructorInfo,
    DriverInfo,
    JolpicaClient,
    RaceResult,
    ScheduleEntry,
)
from f1_rag.tools import TOOL_DEFINITIONS, TOOL_REGISTRY, execute_tool


def test_tool_definitions_match_registry_keys() -> None:
    """Every tool defined for the LLM has an executor; no orphans."""
    def_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    reg_names = set(TOOL_REGISTRY)
    assert def_names == reg_names


@pytest.mark.parametrize("tool", TOOL_DEFINITIONS)
def test_each_tool_definition_has_required_openai_fields(tool: dict[str, Any]) -> None:
    assert tool["type"] == "function"
    fn = tool["function"]
    assert isinstance(fn["name"], str) and fn["name"]
    assert isinstance(fn["description"], str) and len(fn["description"]) > 20
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params


def _fake_driver() -> DriverInfo:
    return DriverInfo(
        driverId="max_verstappen",
        givenName="Max",
        familyName="Verstappen",
        code="VER",
    )


def _fake_constructor() -> ConstructorInfo:
    return ConstructorInfo(constructorId="red_bull", name="Red Bull")


def test_execute_tool_get_race_results_returns_serialisable_dict() -> None:
    client = MagicMock(spec=JolpicaClient)
    client.get_race_results.return_value = [
        RaceResult(
            position=1,
            points=25.0,
            grid=10,
            status="Finished",
            time="1:34:24.258",
            driver=_fake_driver(),
            constructor=_fake_constructor(),
        )
    ]
    result = execute_tool("get_race_results", {"season": 2022, "round": 13}, client)
    client.get_race_results.assert_called_once_with(season=2022, round_=13)
    assert result["season"] == 2022
    assert result["round"] == 13
    assert result["n_results"] == 1
    assert result["results"][0]["driver"]["family_name"] == "Verstappen"


def test_execute_tool_get_race_schedule_returns_serialisable_dict() -> None:
    client = MagicMock(spec=JolpicaClient)
    client.get_race_schedule.return_value = [
        ScheduleEntry(
            round=13,
            race_name="Hungarian Grand Prix",
            circuit_id="hungaroring",
            circuit_name="Hungaroring",
            locality="Budapest",
            country="Hungary",
            date="2022-07-31",
        )
    ]
    result = execute_tool("get_race_schedule", {"season": 2022}, client)
    assert result["season"] == 2022
    assert len(result["schedule"]) == 1
    assert result["schedule"][0]["race_name"] == "Hungarian Grand Prix"


def test_execute_tool_unknown_name_raises() -> None:
    client = MagicMock(spec=JolpicaClient)
    with pytest.raises(KeyError):
        execute_tool("get_thing_that_does_not_exist", {}, client)
