"""  
Replay gold tactic traces from the dataset through LeanDojo.  
  
Usage:  
    python scripts/replay_proof.py --limit 5  
    python scripts/replay_proof.py --limit 10 --split test --output artifacts/replay/results.json  
"""  
  
from __future__ import annotations  
  
import argparse  
import sys  
from pathlib import Path 
import re  
  
if __package__ in {None, ""}:  
    repo_root = Path(__file__).resolve().parents[1]  
    if str(repo_root) not in sys.path:  
        sys.path.insert(0, str(repo_root))  
  
from datasets import load_dataset  
  
from atp_lean_gnn.dataset import dataset_split_name  
from atp_lean_gnn.lean_env import LeanEnvironment, theorem_info_from_dataset_row  
from atp_lean_gnn.replay import ReplaySummary, replay_proof  
from atp_lean_gnn.reporting import console_print  

def strip_lean_dojo_tags(tactic: str) -> str:  
    """Remove <a>...</a> premise annotations that LeanDojo adds to tactics."""  
    return re.sub(r"</?a[^>]*>", "", tactic)
  
def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser(description="Replay gold tactic traces")  
    parser.add_argument("--limit", type=int, default=5)  
    parser.add_argument("--split", type=str, default="test")  
    parser.add_argument("--output", type=str, default="artifacts/replay/results.json")  
    parser.add_argument("--dataset-name", type=str,  
                        default="cat-searcher/leandojo-benchmark-4-random")  
    args = parser.parse_args(argv)  
  
    hf_split = dataset_split_name(args.split)  
    console_print(f"\n  Loading dataset split={hf_split}...")  
  
    ds = load_dataset(args.dataset_name, split=hf_split, streaming=True)  
  
    # Group rows by theorem to reconstruct full tactic traces.  
    # Each HF row is one tactic step; multiple rows share the same full_name.  
    theorem_traces: dict[str, dict] = {}  
    for row in ds:  
        full_name = row.get("full_name", "")  
        if not full_name:  
            continue  
        if full_name not in theorem_traces:  
            theorem_traces[full_name] = {  
                "info": theorem_info_from_dataset_row(row),  
                "tactics": [],  
            }  
            if len(theorem_traces) > args.limit:  
                break  
        theorem_traces[full_name]["tactics"].append(strip_lean_dojo_tags(row.get("tactic", "")))  
  
    console_print(f"  Collected {len(theorem_traces)} theorem traces")  
  
    summary = ReplaySummary()  
    with LeanEnvironment() as env:  
        for name, data in list(theorem_traces.items())[:args.limit]:  
            console_print(f"\n  Replaying: {name} ({len(data['tactics'])} steps)")  
            try:  
                trace = replay_proof(env, data["info"], data["tactics"])  
                summary.add(trace)  
                status = "OK" if trace.success else f"FAIL step {trace.error_step}"  
                console_print(f"    {status}")  
            except Exception as exc:  
                console_print(f"    ERROR: {exc}")  
  
    out_path = summary.save(args.output)  
    console_print(f"\n  {summary.succeeded}/{summary.total} succeeded")  
    console_print(f"  Saved: {out_path}")  
    return 0  
  
  
if __name__ == "__main__":  
    raise SystemExit(main())