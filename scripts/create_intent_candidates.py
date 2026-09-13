#!/usr/bin/env python
"""Generate candidate intent clusters for a brand from its TRAIN-split
customer messages.

    python scripts/create_intent_candidates.py --brand AmazonHelp
    python scripts/create_intent_candidates.py --brand AmazonHelp --min-k 8 --max-k 15

Writes candidates to `data/processed/intent_candidates_<brand>.json` for
human review. This script never writes `configs/intents.yaml` directly —
finalizing the taxonomy from these candidates is a human step (see the
README's "How intents are discovered" section).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/create_intent_candidates.py` from the
# repo root without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.intents import main

if __name__ == "__main__":
    main()
