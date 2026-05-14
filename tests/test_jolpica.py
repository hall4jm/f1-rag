"""Unit tests for jolpica.py -- offline (cache stubs, no real HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from f1_rag.config import Settings
from f1_rag.jolpica import (
    JolpicaClient,
    _parse_race_result,
    _parse_schedule_entry,
)


def _ergast_results_fixture() -> dict:
    """Minimal Ergast-shaped race results payload for one driver."""
    return {
        "MRData": {
            "RaceTable": {
                "season": "2022",
                "round": "13",
                "Races": [
                    {
                        "season": "2022",
                        "round": "13",
                        "raceName": "Hungarian Grand Prix",
                        "Results": [
                            {
                                "number": "1",
                                "position": "1",
                                "points": "25",
                                "grid": "10",
                                "status": "Finished",
                                "Time": {"time": "1:34:24.258"},
                                "Driver": {
                                    "driverId": "max_verstappen",
                                    "code": "VER",
                                    "givenName": "Max",
                                    "familyName": "Verstappen",
                                },
                                "Constructor": {
                                    "constructorId": "red_bull",
                                    "name": "Red Bull",
                                },
                                "FastestLap": {"rank": "3"},
                            }
                        ],
                    }
                ],
            }
        }
    }


def _make_client(tmp_path: Path) -> JolpicaClient:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    return JolpicaClient(settings=settings, cache_dir=tmp_path / "jolpica_cache")


def test_cache_hit_skips_http(tmp_path: Path) -> None:
    """A pre-existing cache file is read directly; no HTTP call is made."""
    client = _make_client(tmp_path)
    cache_path = client._cache_path("2022/13/results.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_ergast_results_fixture()), encoding="utf-8")

    with patch.object(JolpicaClient, "_fetch_uncached") as mock_fetch:
        results = client.get_race_results(season=2022, round_=13)
        mock_fetch.assert_not_called()

    assert len(results) == 1
    assert results[0].driver.driver_id == "max_verstappen"
    assert results[0].position == 1
    assert results[0].grid == 10


def test_cache_miss_writes_cache(tmp_path: Path) -> None:
    """A cache miss fetches via HTTP and writes the response to disk."""
    client = _make_client(tmp_path)
    payload = _ergast_results_fixture()

    with patch.object(JolpicaClient, "_fetch_uncached", return_value=payload) as mock_fetch:
        client.get_race_results(season=2022, round_=13)
        mock_fetch.assert_called_once_with("2022/13/results.json")

    cache_path = client._cache_path("2022/13/results.json")
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == payload


def test_parse_race_result_extracts_typed_fields() -> None:
    raw = _ergast_results_fixture()["MRData"]["RaceTable"]["Races"][0]["Results"][0]
    parsed = _parse_race_result(raw)
    assert parsed.position == 1
    assert parsed.points == 25.0
    assert parsed.grid == 10
    assert parsed.status == "Finished"
    assert parsed.time == "1:34:24.258"
    assert parsed.driver.code == "VER"
    assert parsed.constructor.name == "Red Bull"
    assert parsed.fastest_lap_rank == 3


def test_parse_race_result_handles_dnf_without_time() -> None:
    raw = {
        "position": "18",
        "points": "0",
        "grid": "12",
        "status": "Retired",
        "Driver": {
            "driverId": "x_driver",
            "givenName": "X",
            "familyName": "Driver",
        },
        "Constructor": {"constructorId": "x", "name": "X"},
    }
    parsed = _parse_race_result(raw)
    assert parsed.time is None
    assert parsed.fastest_lap_rank is None


def test_parse_schedule_entry_extracts_circuit_location() -> None:
    raw = {
        "round": "13",
        "raceName": "Hungarian Grand Prix",
        "date": "2022-07-31",
        "Circuit": {
            "circuitId": "hungaroring",
            "circuitName": "Hungaroring",
            "Location": {"locality": "Budapest", "country": "Hungary"},
        },
    }
    entry = _parse_schedule_entry(raw)
    assert entry.round == 13
    assert entry.race_name == "Hungarian Grand Prix"
    assert entry.circuit_id == "hungaroring"
    assert entry.locality == "Budapest"
    assert entry.country == "Hungary"


def test_empty_races_list_returns_empty(tmp_path: Path) -> None:
    """A season with no matching race (e.g. round beyond schedule) returns []."""
    client = _make_client(tmp_path)
    payload = {"MRData": {"RaceTable": {"Races": []}}}
    with patch.object(JolpicaClient, "_fetch_uncached", return_value=payload):
        assert client.get_race_results(season=2099, round_=99) == []
