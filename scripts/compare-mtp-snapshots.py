#!/usr/bin/env python3
"""Compare bounded LLAMA_MTP_SNAPSHOT_PREFIX captures from CPU and HRX draft contexts."""

import argparse
import array
import csv
import math
from pathlib import Path
import sys


def read_manifest(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        if row["key"] in result:
            raise ValueError(f"duplicate key: {row['key']}")
        result[row["key"]] = row
    return result


def read_values(row, manifest):
    path = Path(row["file"])
    if not path.is_file():
        path = manifest.parent / path.name
    size = path.stat().st_size
    if size > 16 * 1024 * 1024 or size != int(row["count"]) * 4:
        raise ValueError(f"invalid snapshot size: {path}")
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, int(row["count"]))
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="CPU draft .tsv manifest")
    parser.add_argument("actual", type=Path, help="HRX draft .tsv manifest")
    parser.add_argument("--atol", type=float, default=0.0001)
    parser.add_argument("--rtol", type=float, default=0.001)
    args = parser.parse_args()
    reference = read_manifest(args.reference)
    actual = read_manifest(args.actual)
    if not reference or not actual:
        raise ValueError("empty capture: check phase/call limit and named tensor selection")
    first = None
    mismatches = 0
    # Manifest order is execution order; do not sort tensor names alphabetically.
    for key, expected_row in reference.items():
        if key not in actual:
            print(f"MISSING {key}")
            mismatches += 1
            first = first or key
            continue
        row = actual[key]
        shape = tuple(int(row[f"ne{i}"]) for i in range(4))
        expected_shape = tuple(int(expected_row[f"ne{i}"]) for i in range(4))
        if shape != expected_shape or row["count"] != expected_row["count"]:
            print(f"SHAPE {key}: cpu={expected_shape} actual={shape}")
            mismatches += 1
            first = first or key
            continue
        expected = read_values(expected_row, args.reference)
        values = read_values(row, args.actual)
        bad = nonfinite = 0
        max_abs = squares = ref_squares = 0.0
        worst = first_bad = None
        for i, (a, b) in enumerate(zip(values, expected)):
            if not math.isfinite(a) or not math.isfinite(b):
                nonfinite += 1
                bad += 1
                if first_bad is None:
                    first_bad = i
                continue
            difference = abs(a - b)
            squares += difference * difference
            ref_squares += b * b
            if difference > max_abs:
                max_abs, worst = difference, i
            # Inputs must be identical, otherwise downstream comparisons are not an isolation.
            atol, rtol = (0.0, 0.0) if row["op"] == "INPUT" else (args.atol, args.rtol)
            if difference > atol + rtol * abs(b):
                bad += 1
                if first_bad is None:
                    first_bad = i
        relative_l2 = math.sqrt(squares / ref_squares) if ref_squares else math.sqrt(squares)
        print(f"{'DIFF' if bad else 'OK'} {key} cpu={expected_row['buffer']} actual={row['buffer']} "
              f"shape={shape} bad={bad}/{len(values)} nonfinite={nonfinite} "
              f"max_abs={max_abs:.8g} relative_l2={relative_l2:.8g} worst={worst} first_bad={first_bad}")
        if first_bad is not None:
            print(f"  first_bad[{first_bad}] cpu={expected[first_bad]:.9g} actual={values[first_bad]:.9g}")
        if bad:
            first = first or key
            mismatches += 1
    for key in actual.keys() - reference.keys():
        print(f"EXTRA {key}")
        mismatches += 1
    print(f"First differing captured stage: {first or 'none'}")
    print("Differences below numerical tolerances are not proof of correctness; captures change partitions.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
