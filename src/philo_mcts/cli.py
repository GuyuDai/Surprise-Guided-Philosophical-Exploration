from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-1 decoding experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.json"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    output = run(config, args.config.resolve())
    print(f"results: {output}")


if __name__ == "__main__":
    main()

