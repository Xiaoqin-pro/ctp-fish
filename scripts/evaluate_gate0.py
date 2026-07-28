from __future__ import annotations
import argparse
parser=argparse.ArgumentParser(description="Evaluate only a frozen Gate-0 checkpoint on a selected dev split.")
parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--dry-run",action="store_true")
if __name__=="__main__":
    args=parser.parse_args(); print("Evaluation scaffold ready; it will report image and trajectory-balanced metrics.")
