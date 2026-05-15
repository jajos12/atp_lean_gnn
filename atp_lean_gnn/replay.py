"""  
Gold-trace proof replay through the Lean environment.  
"""  
  
from __future__ import annotations  
  
import json  
import logging  
from dataclasses import dataclass, field  
from pathlib import Path  
from typing import Optional  
  
from .lean_env import LeanEnvironment, StepResult, TheoremInfo  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass(frozen=True)  
class ReplayStep:  
    step_index: int  
    tactic: str  
    result: StepResult  
    state_before: str  
  
  
@dataclass  
class ReplayTrace:  
    theorem_info: TheoremInfo  
    initial_state: str  
    steps: list[ReplayStep] = field(default_factory=list)  
    success: bool = False  
    error_step: Optional[int] = None  
    error_message: str = ""  
  
    def to_dict(self) -> dict:  
        return {  
            "theorem": self.theorem_info.full_name,  
            "file_path": self.theorem_info.file_path,  
            "initial_state": self.initial_state,  
            "success": self.success,  
            "num_steps": len(self.steps),  
            "error_step": self.error_step,  
            "error_message": self.error_message,  
            "steps": [  
                {  
                    "step_index": s.step_index,  
                    "tactic": s.tactic,  
                    "success": s.result.success,  
                    "completed": s.result.completed,  
                    "state_before": s.state_before,  
                    "state_after": s.result.state_text,  
                    "error": s.result.error_message,  
                }  
                for s in self.steps  
            ],  
        }  
  
  
def replay_proof(  
    env: LeanEnvironment,  
    theorem_info: TheoremInfo,  
    tactics: list[str],  
) -> ReplayTrace:  
    """Replay a known tactic sequence on a theorem."""  
    initial_state = env.load_theorem(theorem_info)  
    trace = ReplayTrace(theorem_info=theorem_info, initial_state=initial_state)  
  
    for i, tactic in enumerate(tactics):  
        state_before = env.current_state()  
        result = env.apply_tactic(tactic)  
  
        trace.steps.append(ReplayStep(  
            step_index=i, tactic=tactic,  
            result=result, state_before=state_before,  
        ))  
  
        if not result.success:  
            trace.error_step = i  
            trace.error_message = result.error_message  
            return trace  
  
        if result.completed:  
            trace.success = True  
            return trace  
  
    trace.error_message = "All tactics applied but proof not complete"  
    return trace  
  
  
@dataclass  
class ReplaySummary:  
    total: int = 0  
    succeeded: int = 0  
    failed: int = 0  
    traces: list[ReplayTrace] = field(default_factory=list)  
  
    def add(self, trace: ReplayTrace) -> None:  
        self.total += 1  
        if trace.success:  
            self.succeeded += 1  
        else:  
            self.failed += 1  
        self.traces.append(trace)  
  
    def success_rate(self) -> float:  
        return self.succeeded / self.total if self.total > 0 else 0.0  
  
    def save(self, path: str | Path) -> Path:  
        out = Path(path)  
        out.parent.mkdir(parents=True, exist_ok=True)  
        out.write_text(json.dumps({  
            "total": self.total,  
            "succeeded": self.succeeded,  
            "failed": self.failed,  
            "success_rate": self.success_rate(),  
            "traces": [t.to_dict() for t in self.traces],  
        }, indent=2, ensure_ascii=False), encoding="utf-8")  
        return out