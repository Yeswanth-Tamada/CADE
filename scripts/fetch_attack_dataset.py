"""Download a pinned MITRE ATT&CK Enterprise STIX bundle (run manually; never at app startup).

Usage (from the project root, with the virtual environment active)::

    python scripts/fetch_attack_dataset.py                 # default pinned version
    python scripts/fetch_attack_dataset.py --version 19.2  # another release
    python scripts/fetch_attack_dataset.py --output path/to/enterprise-attack.json

Source: https://github.com/mitre-attack/attack-stix-data (official MITRE repository).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.intent.attack_mapper import AttackDataset, AttackDatasetError  # noqa: E402

DEFAULT_VERSION = "20.0"
URL_TEMPLATE = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack-{version}.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "app" / "data" / "attack" / "enterprise-attack.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION, help=f"ATT&CK release (default {DEFAULT_VERSION})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    url = URL_TEMPLATE.format(version=args.version)
    tmp = args.output.with_suffix(".download")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ATT&CK Enterprise v{args.version}\n  from {url}")
    try:
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    try:
        dataset = AttackDataset.from_stix_file(tmp)
    except AttackDatasetError as exc:
        tmp.unlink(missing_ok=True)
        print(f"Downloaded file is not a usable ATT&CK bundle: {exc}", file=sys.stderr)
        return 1

    tmp.replace(args.output)
    active = sum(1 for t in dataset.techniques.values() if t.active)
    print(f"Saved to {args.output}")
    print(f"ATT&CK version: {dataset.version}; techniques: {len(dataset.techniques)} ({active} active)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
