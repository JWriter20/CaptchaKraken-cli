#!/usr/bin/env python3
"""Run the Move-indicator detector over a folder of images and report counts.

Used to validate `find_move_indicators` against the golden cleanSamples store
(or any image dir): prints how many Move pills each image contains, a per-folder
summary, and a total. Default root is the parent repo's cleanSamples/test/raw.

  python scripts/sweep_move_indicators.py [ROOT_DIR] [--all]

By default only images WITH at least one indicator are listed; pass --all to
list every scanned image (including 0-count).
"""

import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tool_calls.move_indicator import find_move_indicators  # noqa: E402

_DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "cleanSamples", "test", "raw")
)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--all"]
    show_all = "--all" in sys.argv
    root = args[0] if args else _DEFAULT_ROOT
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rows = []
    by_folder = defaultdict(lambda: [0, 0, 0])  # folder -> [imgs_with, total, pills]
    scanned = 0
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(dirpath, fn)
            im = cv2.imread(path)
            if im is None:
                continue
            scanned += 1
            n = len(find_move_indicators(im))
            rel = os.path.relpath(path, root)
            folder = rel.split(os.sep)[0]
            by_folder[folder][1] += 1
            by_folder[folder][2] += n
            if n:
                by_folder[folder][0] += 1
                rows.append((n, rel))
            elif show_all:
                rows.append((0, rel))

    rows.sort(key=lambda r: (-r[0], r[1]))
    for n, rel in rows:
        print(f"{n}\t{rel}")

    total_pills = sum(v[2] for v in by_folder.values())
    imgs_with = sum(v[0] for v in by_folder.values())
    print("\n--- per folder (imgs_with / total : pills) ---")
    for folder in sorted(by_folder):
        w, t, pl = by_folder[folder]
        print(f"  {w:4d}/{t:<5d} {pl:4d}  {folder}")
    print(f"\nScanned {scanned} images; {imgs_with} had >=1 Move indicator "
          f"({total_pills} pills total).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
