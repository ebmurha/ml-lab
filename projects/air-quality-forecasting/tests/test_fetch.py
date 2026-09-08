from datetime import datetime, timedelta, timezone

import httpx
import pytest

from air_quality_forecasting.fetch import (
    MAX_RANGE,
    OpenAQClient,
    fetch_location,
    iter_ranges,
    parse_utc,
)


def test_ranges_are_bounded_and_contiguous() -> None:
    start = parse_utc("2024-01-01T00:00:00Z")
    end = parse_utc("2026-01-02T00:00:00Z")

    ranges = list(iter_ranges(start, end))

    assert ranges[0][0] == start
    assert ranges[-1][1] == end
    assert all(right - left <= MAX_RANGE for left, right in ranges)
    assert all(current[1] == following[0] for current, following in zip(ranges, ranges[1:]))


def test_parse_utc_rejects_naive_datetime() -> None:
    with pytest.raises(Exception, match="timezone"):
        parse_utc("2025-01-01T00:00:00")


def test_fetches_all_pages_and_deduplicates_chunk_boundaries() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/sensors"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 7,
                            "name": "pm25 sensor",
                            "parameter": {"name": "pm25", "units": "µg/m³"},
                        }
                    ]
                },
            )
        page = int(request.url.params["page"])
        if page == 1:
            results = [measurement(index) for index in range(1_000)]
        elif page == 2:
            results = [measurement(999), measurement(1_000)]
        else:
            results = []
        return httpx.Response(200, json={"meta": {"found": 1_001}, "results": results})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="https://api.openaq.org/v3", transport=transport)
    api = OpenAQClient("not-a-secret", client=http)

    rows = fetch_location(
        api,
        42,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 2, 1, tzinfo=timezone.utc),
        {"pm25"},
    )

    assert len(rows) == 1_001
    assert rows[0]["location_id"] == 42
    assert len(calls) == 3
    http.close()


def measurement(hour: int) -> dict[str, object]:
    observed_at = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)
    timestamp = observed_at.isoformat().replace("+00:00", "Z")
    return {
        "value": float(hour),
        "parameter": {"name": "pm25", "units": "µg/m³"},
        "period": {
            "datetimeFrom": {"utc": timestamp, "local": timestamp},
            "datetimeTo": {"utc": timestamp, "local": timestamp},
        },
        "coordinates": {"latitude": -1.0, "longitude": 36.0},
        "flagInfo": {"hasFlags": False},
    }
