from __future__ import annotations
import argparse
parser=argparse.ArgumentParser(description="Run the fixed online mask-view background-correlation audit after official masks are matched.")
parser.add_argument("--config",required=True); parser.add_argument("--dry-run",action="store_true")
if __name__=="__main__":
    args=parser.parse_args(); print("Background audit is disabled until official masks are audited and matched.")
