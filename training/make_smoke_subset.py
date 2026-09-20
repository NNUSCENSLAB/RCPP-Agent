#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    lines = [line for line in args.source.read_text(encoding="utf-8").splitlines() if line]
    args.output.write_text("\n".join(lines[: args.count]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
