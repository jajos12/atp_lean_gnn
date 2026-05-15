"""  
Proof search loop: GNN tactic prediction + LeanDojo environment stepping.  
"""  
  
from __future__ import annotations  
  
import json  
import logging  
from dataclasses import dataclass, field  
from pathlib import Path  
  
import torch  
  
from .graph import proof_state_to_dag  
from .lean_env import LeanEnvironment, StepResult, TheoremInfo  
from .model import GraphSAGEStateClassifier  
from .pyg import dag_to_pyg  
from .state import parse_state  
from .training import PreparedMetadata, infer_state_node_index, transform_edge_index  
  
logger = logging.getLogger(__name__)  
  
  
# ── Config ────────────────────────────────────────────────────────────  
  
@dataclass(frozen=True)  
class SearchConfig:  
    beam_width: int = 5       # top-k tactics to try per step  
    max_depth: int = 50       # max proof steps  
    edge_mode: str = "bidirectional"  
  
  
# ── Trace types ───────────────────────────────────────────────────────  
  
@dataclass  
class SearchStep:  
    depth: int  
    state_text: str  
    tactic_tried: str  
    tactic_rank: int  
    probability: float  
    success: bool  
    completed: bool  
    error_message: str = ""  
  
  
@dataclass  
class SearchTrace:  
    theorem_info: TheoremInfo  
    initial_state: str  
    steps: list[SearchStep] = field(default_factory=list)  
    solved: bool = False  
    final_depth: int = 0  
    failure_reason: str = ""  
  
    def to_dict(self) -> dict:  
        return {  
            "theorem": self.theorem_info.full_name,  
            "file_path": self.theorem_info.file_path,  
            "solved": self.solved,  
            "final_depth": self.final_depth,  
            "failure_reason": self.failure_reason,  
            "num_steps_tried": len(self.steps),  
            "steps": [  
                {  
                    "depth": s.depth,  
                    "tactic": s.tactic_tried,  
                    "rank": s.tactic_rank,  
                    "probability": s.probability,  
                    "success": s.success,  
                    "completed": s.completed,  
                    "error": s.error_message,  
                }  
                for s in self.steps  
            ],  
        }  
  
  
# ── Bridge: live state text → PyG graph ──────────────────────────────  
  
def state_text_to_pyg(  
    state_text: str,  
    metadata: PreparedMetadata,  
    *,  
    edge_mode: str = "bidirectional",  
    device: torch.device = torch.device("cpu"),  
):  
    """  
    Convert a live proof state string (from LeanDojo) into a PyG Data  
    object ready for the GNN.  
  
    Reuses the existing offline pipeline:  
      state text → parse_state() → proof_state_to_dag() → dag_to_pyg()  
    Then adds the fields the model expects (state_node_index, edge transforms).  
    """  
    parsed = parse_state(state_text)  
    dag = proof_state_to_dag(parsed)  
    data = dag_to_pyg(dag, metadata.node_vocab)  
  
    data.x = data.x.to(dtype=torch.long)  
    data.node_type = data.node_type.to(dtype=torch.long)  
    data.edge_index = transform_edge_index(data.edge_index, edge_mode=edge_mode)  
    data.state_node_index = infer_state_node_index(  
        data,  
        state_label_id=metadata.state_label_id,  
        path=Path("<live>"),  
    )  
  
    return data.to(device)  
  
  
# ── Tactic prediction ────────────────────────────────────────────────  
  
def predict_tactics(  
    model: GraphSAGEStateClassifier,  
    data,  
    *,  
    tactic_vocab: dict[str, int],  
    top_k: int = 5,  
) -> list[tuple[str, float]]:  
    """  
    Run the GNN on a proof state graph and return top-k  
    predicted tactic names with probabilities.  
    """  
    id_to_tactic = {v: k for k, v in tactic_vocab.items()}  
  
    model.eval()  
    with torch.no_grad():  
        logits = model(data)                    # [1, num_tactics]  
        probs = torch.softmax(logits, dim=-1)  
        top_probs, top_indices = probs.topk(top_k, dim=-1)  
  
    results = []  
    for prob, idx in zip(top_probs[0], top_indices[0]):  
        name = id_to_tactic.get(int(idx.item()), "<UNK>")  
        results.append((name, float(prob.item())))  
    return results  
  
  
# ── Search loop ───────────────────────────────────────────────────────  
  
def greedy_search(  
    model: GraphSAGEStateClassifier,  
    env: LeanEnvironment,  
    theorem_info: TheoremInfo,  
    metadata: PreparedMetadata,  
    config: SearchConfig,  
    *,  
    device: torch.device = torch.device("cpu"),  
) -> SearchTrace:  
    """  
    Greedy proof search: at each step, try top-k predicted tactics  
    and take the first one that succeeds.  
  
    IMPORTANT: The model predicts tactic HEADS only (e.g. 'simp'),  
    not full tactics with arguments (e.g. 'rw [h]'). Only argument-free  
    tactics will actually work until premise selection is added.  
    """  
    initial_state = env.load_theorem(theorem_info)  
    trace = SearchTrace(theorem_info=theorem_info, initial_state=initial_state)  
  
    for depth in range(config.max_depth):  
        state_text = env.current_state()  
  
        # Convert live state → PyG graph  
        try:  
            data = state_text_to_pyg(  
                state_text, metadata,  
                edge_mode=config.edge_mode, device=device,  
            )  
        except Exception as exc:  
            trace.failure_reason = f"Graph conversion failed at depth {depth}: {exc}"  
            break  
  
        # Get model predictions  
        candidates = predict_tactics(  
            model, data,  
            tactic_vocab=metadata.tactic_vocab,  
            top_k=config.beam_width,  
        )  
  
        # Try each candidate tactic  
        step_succeeded = False  
        for rank, (tactic_name, prob) in enumerate(candidates):  
            result = env.apply_tactic(tactic_name)  
  
            trace.steps.append(SearchStep(  
                depth=depth, state_text=state_text,  
                tactic_tried=tactic_name, tactic_rank=rank,  
                probability=prob,  
                success=result.success, completed=result.completed,  
                error_message=result.error_message,  
            ))  
  
            if result.completed:  
                trace.solved = True  
                trace.final_depth = depth + 1  
                return trace  
  
            if result.success:  
                step_succeeded = True  
                break  
  
        if not step_succeeded:  
            trace.failure_reason = (  
                f"All {config.beam_width} candidates failed at depth {depth}"  
            )  
            trace.final_depth = depth  
            break  
  
    if not trace.solved and not trace.failure_reason:  
        trace.failure_reason = f"Max depth {config.max_depth} reached"  
        trace.final_depth = config.max_depth  
  
    return trace