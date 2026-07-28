"""Reserved Gate-0 nearest-neighbour audit. Never changes class eligibility or splits."""
from __future__ import annotations
import argparse
parser=argparse.ArgumentParser(description="Compute fixed pHash and ResNet18 nearest-neighbour audit after real data is available.")
parser.add_argument("--metadata",required=True); parser.add_argument("--image-split",required=True); parser.add_argument("--track-split",required=True); parser.add_argument("--dry-run",action="store_true")
if __name__=="__main__":
    args=parser.parse_args(); print("Similarity audit is scaffolded; run only after successful data audit and deterministic splits.")
