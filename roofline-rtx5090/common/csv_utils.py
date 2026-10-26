"""Append-only CSV writer shared by every benchmark script, so results from
repeated/partial runs accumulate in one file instead of overwriting."""
import csv
import os


def append_row(path, row: dict):
    """Append `row` to the CSV at `path`, writing the header from `row`'s
    keys the first time the file is created. All rows written to the same
    file must share the same keys.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
