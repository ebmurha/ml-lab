"""Retrieve paginated OpenAQ measurements for a location."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

API_BASE_URL = "https://api.openaq.org/v3"
PAGE_LIMIT = 1_000
MAX_RANGE = timedelta(days=365)
FIELDNAMES = (
    "location_id",
    "sensor_id",
    "sensor_name",
    "parameter",
    "units",
    "value",
    "datetime_utc_start",
    "datetime_utc_end",
    "datetime_local_start",
    "datetime_local_end",
    "latitude",
    "longitude",
    "has_flags",
)


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 datetime and normalize it to UTC."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone")
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Format a datetime using OpenAQ's UTC representation."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def iter_ranges(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Yield adjacent API ranges no longer than one year."""
    if start >= end:
        raise ValueError("datetime-from must be earlier than datetime-to")
    cursor = start
    while cursor < end:
        boundary = min(cursor + MAX_RANGE, end)
        yield cursor, boundary
        cursor = boundary


class OpenAQClient:
    """Small OpenAQ v3 client with bounded retries."""

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=API_BASE_URL,
            headers={"X-API-Key": api_key},
            timeout=30,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(4):
            response = self._client.get(path, params=params)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response.json()
            if attempt == 3:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(min(delay, 30))
        raise RuntimeError("OpenAQ request failed")

    def sensors(self, location_id: int) -> list[dict[str, Any]]:
        payload = self._get(f"/locations/{location_id}/sensors")
        return list(payload.get("results", []))

    def measurements(
        self,
        sensor_id: int,
        start: datetime,
        end: datetime,
    ) -> Iterator[dict[str, Any]]:
        page = 1
        while True:
            payload = self._get(
                f"/sensors/{sensor_id}/measurements",
                params={
                    "datetime_from": iso_utc(start),
                    "datetime_to": iso_utc(end),
                    "limit": PAGE_LIMIT,
                    "page": page,
                },
            )
            results = list(payload.get("results", []))
            yield from results
            if len(results) < PAGE_LIMIT:
                break
            page += 1


def select_sensors(
    sensors: Iterable[dict[str, Any]], parameters: set[str] | None
) -> list[dict[str, Any]]:
    selected = [
        sensor
        for sensor in sensors
        if parameters is None or sensor["parameter"]["name"].lower() in parameters
    ]
    return sorted(selected, key=lambda sensor: int(sensor["id"]))


def flatten_measurement(
    location_id: int,
    sensor: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    period = measurement.get("period") or {}
    start = period.get("datetimeFrom") or {}
    end = period.get("datetimeTo") or {}
    coordinates = measurement.get("coordinates") or {}
    flag_info = measurement.get("flagInfo") or {}
    parameter = measurement.get("parameter") or sensor["parameter"]
    return {
        "location_id": location_id,
        "sensor_id": sensor["id"],
        "sensor_name": sensor["name"],
        "parameter": parameter.get("name"),
        "units": parameter.get("units"),
        "value": measurement.get("value"),
        "datetime_utc_start": start.get("utc"),
        "datetime_utc_end": end.get("utc"),
        "datetime_local_start": start.get("local"),
        "datetime_local_end": end.get("local"),
        "latitude": coordinates.get("latitude"),
        "longitude": coordinates.get("longitude"),
        "has_flags": flag_info.get("hasFlags", False),
    }


def fetch_location(
    api: OpenAQClient,
    location_id: int,
    start: datetime,
    end: datetime,
    parameters: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch and deduplicate measurements for all selected location sensors."""
    sensors = select_sensors(api.sensors(location_id), parameters)
    if not sensors:
        requested = ", ".join(sorted(parameters)) if parameters else "any parameter"
        raise ValueError(f"location {location_id} has no sensors for {requested}")

    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sensor in sensors:
        for chunk_start, chunk_end in iter_ranges(start, end):
            for measurement in api.measurements(sensor["id"], chunk_start, chunk_end):
                row = flatten_measurement(location_id, sensor, measurement)
                key = (
                    row["sensor_id"],
                    row["datetime_utc_start"],
                    row["datetime_utc_end"],
                )
                rows[key] = row
    return sorted(
        rows.values(),
        key=lambda row: (row["datetime_utc_start"] or "", int(row["sensor_id"])),
    )


def write_csv(rows: Sequence[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--datetime-from", type=parse_utc, required=True)
    parser.add_argument("--datetime-to", type=parse_utc, required=True)
    parser.add_argument("--parameters", nargs="+", type=str.lower)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    api_key = os.getenv("OPENAQ_API_KEY") or os.getenv("OPENAQ-API-KEY")
    if not api_key:
        raise SystemExit(
            f"OpenAQ API key not found; set OPENAQ-API-KEY in {args.env_file}"
        )

    output = args.output or Path("data") / (
        f"openaq_location_{args.location_id}_"
        f"{args.datetime_from.date()}_{args.datetime_to.date()}.csv"
    )
    api = OpenAQClient(api_key)
    try:
        rows = fetch_location(
            api,
            args.location_id,
            args.datetime_from,
            args.datetime_to,
            set(args.parameters) if args.parameters else None,
        )
    finally:
        api.close()

    if not rows:
        raise SystemExit("OpenAQ returned no measurements for the selected range")
    write_csv(rows, output)
    print(f"Saved {len(rows):,} measurements to {output}")


if __name__ == "__main__":
    main()
