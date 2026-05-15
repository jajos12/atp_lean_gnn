"""  
Run proof search: GNN predictions + LeanDojo execution.  
  
Usage:  
    python scripts/run_search.py \  
        --run-dir runs/baseline_gnn/run_XXXXXXXX_XXXXXX \  
        --limit 10  
"""  
  
from __future__ import annotations  
  
import argparse  
import json  
import sys  
from pathlib import Path  
  
if __package__ in {None, ""}:  
    repo_root = Path(__file__).resolve().parents[1]  
    if str(repo_root) not in sys.path:  
        sys.path.insert(0, str(repo_root))  
  
import torch  
from datasets import load_dataset  
  
from atp_lean_gnn.dataset import dataset_split_name  
from atp_lean_gnn.lean_env import LeanEnvironment, theorem_info_from_dataset_row  
from atp_lean_gnn.reporting import console_print  
from atp_lean_gnn.search import SearchConfig, greedy_search  
from atp_lean_gnn.training import (  
    build_model,  
    load_baseline_config,  
    load_prepared_metadata,  
    resolve_device,  
)  
  
  
def main(argv: list[str] | None = None) -> int:  
    parser = argparse.ArgumentParser(description="Proof search with GNN + LeanDojo")  
    parser.add_argument("--run-dir", type=str, required=True,  
                        help="Path to a trained run directory (contains best.pt)")  
    parser.add_argument("--limit", type=int, default=5)  
    parser.add_argument("--beam-width", type=int, default=5)  
    parser.add_argument("--max-depth", type=int, default=50)  
    parser.add_argument("--split", type=str, default="test")  
    parser.add_argument("--output", type=str, default=None)  
    args = parser.parse_args(argv)  
  
    run_dir = Path(args.run_dir)  
  
    # Load config and metadata from the training run  
    config = load_baseline_config(run_dir / "config.json")  
    metadata = load_prepared_metadata(config.prepared_root)  
    device = resolve_device(config.device)  
  
    # Load trained model  
    model = build_model(metadata, config).to(device)  
    checkpoint = torch.load(  
        run_dir / "best.pt", map_location=device, weights_only=False,  
    )  
    model.load_state_dict(checkpoint["model_state_dict"])  
    model.eval()  
    console_print(f"  Loaded model from {run_dir / 'best.pt'}")  
  
    search_config = SearchConfig(  
        beam_width=args.beam_width,  
        max_depth=args.max_depth,  
        edge_mode=config.edge_mode,  
    )  
  
    # Load unique theorems from dataset  
    hf_split = dataset_split_name(args.split)  
    ds = load_dataset(  
        "cat-searcher/leandojo-benchmark-4-random",  
        split=hf_split, streaming=True,  
    )  
    seen: set[str] = set()  
    theorem_list = []  
    for row in ds:  
        name = row.get("full_name", "")  
        if name and name not in seen:  
            seen.add(name)  
            theorem_list.append(theorem_info_from_dataset_row(row))  
        if len(theorem_list) >= args.limit:  
            break  
  
    # Run search  
    results = []  
    solved_count = 0  
    with LeanEnvironment() as env:  
        for i, info in enumerate(theorem_list):  
            console_print(f"\n  [{i+1}/{len(theorem_list)}] {info.full_name}")  
            try:  
                trace = greedy_search(  
                    model, env, info, metadata, search_config,  
                    device=device,  
                )  
                results.append(trace.to_dict())  
                if trace.solved:  
                    solved_count += 1  
                    console_print(f"    SOLVED in {trace.final_depth} steps")  
                else:  
                    console_print(f"    FAILED: {trace.failure_reason}")  
            except Exception as exc:  
                console_print(f"    ERROR: {exc}")  
                results.append({"theorem": info.full_name, "error": str(exc)})  
  
    # Save  
    output_path = Path(args.output or f"artifacts/search/results_{args.split}.json")  
    output_path.parent.mkdir(parents=True, exist_ok=True)  
    output_path.write_text(json.dumps({  
        "total": len(theorem_list),  
        "solved": solved_count,  
        "success_rate": solved_count / max(len(theorem_list), 1),  
        "config": {"beam_width": search_config.beam_width, "max_depth": search_config.max_depth},  
        "results": results,  
    }, indent=2), encoding="utf-8")  
  
    console_print(f"\n  {solved_count}/{len(theorem_list)} solved")  
    console_print(f"  Saved: {output_path}")  
    return 0  
  
  
if __name__ == "__main__":  
    raise SystemExit(main())