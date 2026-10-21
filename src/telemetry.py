"""Parse optional vehicle telemetry and align it to saved video evidence."""

import csv
from io import StringIO
from math import isfinite


TELEMETRY_DISCLAIMER = (
    "Telemetry is optional supplementary context. Time alignment does not prove "
    "which vehicle produced a manoeuvre, establish causation, or assign fault."
)


def parse_telemetry_csv(text, time_column="time_seconds"):
    """Parse a simple CSV with a numeric video-aligned time column.

    Other numeric columns (for example speed, brake, throttle, or steering)
    are preserved exactly as supplied; no vehicle-specific schema is assumed.
    """
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames or time_column not in reader.fieldnames:
        raise ValueError(f"Telemetry CSV must contain a '{time_column}' column.")
    rows = []
    for row_number, row in enumerate(reader, start=2):
        try:
            time_seconds = float(row[time_column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Telemetry row {row_number} has an invalid time value.") from error
        if not isfinite(time_seconds):
            raise ValueError(f"Telemetry row {row_number} has a non-finite time value.")
        values = {}
        for key, value in row.items():
            if key == time_column or value in (None, ""):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if isfinite(number):
                values[key] = number
        rows.append({"time_seconds": time_seconds, "values": values})
    if not rows:
        raise ValueError("Telemetry CSV contains no data rows.")
    return sorted(rows, key=lambda row: row["time_seconds"])


def correlate_telemetry_events(events, telemetry_rows, maximum_time_delta_seconds=0.5):
    """Attach only the nearest telemetry sample inside a transparent time window."""
    if maximum_time_delta_seconds < 0:
        raise ValueError("Maximum telemetry time delta cannot be negative.")
    correlations = []
    for event in events:
        closest = min(
            telemetry_rows,
            key=lambda row: abs(row["time_seconds"] - float(event["time_seconds"])),
        )
        delta = abs(closest["time_seconds"] - float(event["time_seconds"]))
        correlations.append({
            "event_key": event.get("event_key"),
            "event_time_seconds": float(event["time_seconds"]),
            "telemetry_time_seconds": closest["time_seconds"] if delta <= maximum_time_delta_seconds else None,
            "time_delta_seconds": delta,
            "telemetry_values": closest["values"] if delta <= maximum_time_delta_seconds else {},
            "alignment_status": "aligned" if delta <= maximum_time_delta_seconds else "withheld_outside_time_window",
        })
    return correlations
