"""Typed client for the Jolpica F1 API (Ergast-compatible).

Jolpica is the community-run successor to Ergast and serves the same JSON
shape at ``https://api.jolpi.ca/ergast/f1/``. We model only the fields the
v2 agent actually needs -- raw Ergast responses are deeply nested with many
fields the LLM doesn't care about.

Caching: F1 historical data is immutable, so we cache responses to disk
forever in ``data/jolpica_cache/`` (keyed by URL path hash). Clear the
directory if you ever need a fresh fetch.

Rate limit: 200 requests / hour unauthenticated. Tenacity retries on
transient HTTP errors with exponential backoff; cache hits never count
against the limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from f1_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------- typed response models (only the fields the LLM uses) ----------


class DriverInfo(BaseModel):
    """Compact driver record."""

    driver_id: str = Field(alias="driverId")
    code: str | None = None
    given_name: str = Field(alias="givenName")
    family_name: str = Field(alias="familyName")
    date_of_birth: str | None = Field(default=None, alias="dateOfBirth")
    nationality: str | None = None

    model_config = {"populate_by_name": True}


class ConstructorInfo(BaseModel):
    """Compact constructor record."""

    constructor_id: str = Field(alias="constructorId")
    name: str
    nationality: str | None = None

    model_config = {"populate_by_name": True}


class RaceResult(BaseModel):
    """One driver's finishing data for a race."""

    position: int
    points: float
    grid: int
    status: str
    time: str | None = None  # 'X laps' / 'XX:XX.XXX' / None for DNF
    driver: DriverInfo
    constructor: ConstructorInfo
    fastest_lap_rank: int | None = None


class QualifyingResult(BaseModel):
    """One driver's qualifying performance."""

    position: int
    driver: DriverInfo
    constructor: ConstructorInfo
    q1: str | None = None
    q2: str | None = None
    q3: str | None = None


class DriverStanding(BaseModel):
    """One row of a Drivers' Championship table."""

    position: int
    points: float
    wins: int
    driver: DriverInfo
    constructors: list[ConstructorInfo]


class ConstructorStanding(BaseModel):
    """One row of a Constructors' Championship table."""

    position: int
    points: float
    wins: int
    constructor: ConstructorInfo


class ScheduleEntry(BaseModel):
    """One race in a season's calendar."""

    round: int
    race_name: str
    circuit_id: str
    circuit_name: str
    locality: str | None = None
    country: str | None = None
    date: str  # YYYY-MM-DD


# ---------- client ----------


class JolpicaClient:
    """Minimal Jolpica/Ergast client with disk caching and retry.

    Public methods all return typed Pydantic models (lists where appropriate).
    Raw nested Ergast JSON is parsed once at the client boundary so callers
    never see ``MRData['RaceTable']['Races'][0]['Results']`` indexing.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.jolpica_base_url.rstrip("/")
        self.cache_dir = cache_dir or Path("data/jolpica_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- transport -----

    def _cache_path(self, path: str) -> Path:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _fetch_uncached(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.info("jolpica GET %s", url)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def _get(self, path: str) -> dict[str, Any]:
        """Cached GET. F1 historical data is immutable, so cache is forever."""
        cache_path = self._cache_path(path)
        if cache_path.exists():
            logger.debug("jolpica cache hit %s", path)
            return json.loads(cache_path.read_text(encoding="utf-8"))
        data = self._fetch_uncached(path)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    # ----- public methods -----

    def get_race_results(self, season: int, round_: int) -> list[RaceResult]:
        """Final finishing order of one race."""
        data = self._get(f"{season}/{round_}/results.json")
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return []
        return [_parse_race_result(r) for r in races[0]["Results"]]

    def get_qualifying(self, season: int, round_: int) -> list[QualifyingResult]:
        """Qualifying-session results for one race."""
        data = self._get(f"{season}/{round_}/qualifying.json")
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return []
        return [_parse_qualifying(q) for q in races[0]["QualifyingResults"]]

    def get_driver_standings(self, season: int) -> list[DriverStanding]:
        """Final (or current) Drivers' Championship standings for a season."""
        data = self._get(f"{season}/driverStandings.json")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        return [_parse_driver_standing(s) for s in lists[0]["DriverStandings"]]

    def get_constructor_standings(self, season: int) -> list[ConstructorStanding]:
        """Final (or current) Constructors' Championship standings for a season."""
        data = self._get(f"{season}/constructorStandings.json")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return []
        return [_parse_constructor_standing(s) for s in lists[0]["ConstructorStandings"]]

    def get_race_schedule(self, season: int) -> list[ScheduleEntry]:
        """Round-by-round calendar for a season."""
        data = self._get(f"{season}.json")
        races = data["MRData"]["RaceTable"]["Races"]
        return [_parse_schedule_entry(r) for r in races]

    def get_driver_info(self, driver_id: str) -> DriverInfo | None:
        """Biographical info for one driver. Returns None if the id is unknown.

        Note: Ergast driver IDs are slugs like ``max_verstappen`` or
        ``lewis_hamilton`` -- the LLM has to know or guess these.
        """
        data = self._get(f"drivers/{driver_id}.json")
        drivers = data["MRData"]["DriverTable"]["Drivers"]
        if not drivers:
            return None
        return DriverInfo.model_validate(drivers[0])


# ---------- parsers ----------


def _parse_race_result(r: dict[str, Any]) -> RaceResult:
    fastest = r.get("FastestLap")
    return RaceResult(
        position=int(r["position"]),
        points=float(r["points"]),
        grid=int(r["grid"]),
        status=r["status"],
        time=(r.get("Time") or {}).get("time"),
        driver=DriverInfo.model_validate(r["Driver"]),
        constructor=ConstructorInfo.model_validate(r["Constructor"]),
        fastest_lap_rank=int(fastest["rank"]) if fastest and "rank" in fastest else None,
    )


def _parse_qualifying(q: dict[str, Any]) -> QualifyingResult:
    return QualifyingResult(
        position=int(q["position"]),
        driver=DriverInfo.model_validate(q["Driver"]),
        constructor=ConstructorInfo.model_validate(q["Constructor"]),
        q1=q.get("Q1"),
        q2=q.get("Q2"),
        q3=q.get("Q3"),
    )


def _parse_driver_standing(s: dict[str, Any]) -> DriverStanding:
    return DriverStanding(
        position=int(s["position"]),
        points=float(s["points"]),
        wins=int(s["wins"]),
        driver=DriverInfo.model_validate(s["Driver"]),
        constructors=[ConstructorInfo.model_validate(c) for c in s["Constructors"]],
    )


def _parse_constructor_standing(s: dict[str, Any]) -> ConstructorStanding:
    return ConstructorStanding(
        position=int(s["position"]),
        points=float(s["points"]),
        wins=int(s["wins"]),
        constructor=ConstructorInfo.model_validate(s["Constructor"]),
    )


def _parse_schedule_entry(r: dict[str, Any]) -> ScheduleEntry:
    circuit = r["Circuit"]
    location = circuit.get("Location", {})
    return ScheduleEntry(
        round=int(r["round"]),
        race_name=r["raceName"],
        circuit_id=circuit["circuitId"],
        circuit_name=circuit["circuitName"],
        locality=location.get("locality"),
        country=location.get("country"),
        date=r["date"],
    )
