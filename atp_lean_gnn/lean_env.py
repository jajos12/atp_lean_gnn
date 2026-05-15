"""  
Lean environment wrapper using LeanDojo.  
  
Provides: load theorem, get state, apply tactic, detect completion.  
"""  
  
from __future__ import annotations  
  
import logging  
from dataclasses import dataclass  
from typing import Optional  
  
from lean_dojo import (  
    Dojo,  
    LeanGitRepo,  
    ProofFinished,  
    TacticState,  
    Theorem,  
)  
  
# These error classes live in a submodule in lean-dojo v4.20.0  
try:  
    from lean_dojo.interaction.dojo import (  
        DojoTacticTimeoutError,  
        DojoInitError,  
        DojoCrashError,  
    )  
except ImportError:  
    DojoTacticTimeoutError = Exception  
    DojoInitError = Exception  
    DojoCrashError = Exception  
  

try:  
    from lean_dojo import LeanError  
except ImportError:  
    LeanError = Exception 
  
logger = logging.getLogger(__name__)  
  
  
@dataclass(frozen=True)  
class StepResult:  
    """Result of applying one tactic."""  
    success: bool  
    state_text: str  
    completed: bool  
    error_message: str  
    tactic: str  
  
  
@dataclass(frozen=True)  
class TheoremInfo:  
    """Everything LeanDojo needs to load a theorem."""  
    repo_url: str  
    commit: str  
    file_path: str       # e.g. "Mathlib/Data/Nat/Basic.lean"  
    full_name: str        # e.g. "Nat.succ_ne_zero"  
  
  
class LeanEnvironment:  
    """  
    Interactive Lean environment backed by LeanDojo.  
  
    Usage::  
  
        with LeanEnvironment() as env:  
            state = env.load_theorem(info)  
            result = env.apply_tactic("simp")  
            if result.completed:  
                print("Proof done!")  
    """  
  
    def __init__(self) -> None:  
        self._dojo: Optional[Dojo] = None  
        self._current_state: Optional[TacticState] = None  
        self._theorem_info: Optional[TheoremInfo] = None  
        self._completed: bool = False  
        self._dojo_context = None  
  
    def load_theorem(self, info: TheoremInfo) -> str:  
        """Load a theorem and return the initial proof state text."""  
        self.close()  
  
        repo = LeanGitRepo(info.repo_url, info.commit)  
        theorem = Theorem(repo, info.file_path, info.full_name)  
  
        self._dojo_context = Dojo(theorem, timeout=1200)  # 20 min timeout for loading/initialization
        dojo, initial_state = self._dojo_context.__enter__()  
        self._dojo = dojo  
        self._current_state = initial_state  
        self._theorem_info = info  
        self._completed = False  
  
        state_text = str(initial_state.pp)  
        logger.info("Loaded %s — state: %s", info.full_name, state_text[:200])  
        return state_text  
  
    def current_state(self) -> str:  
        """Return current proof state as text."""  
        if self._current_state is None:  
            raise RuntimeError("No theorem loaded. Call load_theorem() first.")  
        return str(self._current_state.pp)  
  
    def is_complete(self) -> bool:  
        return self._completed  
  
    def apply_tactic(self, tactic: str) -> StepResult:  
        """Apply a tactic string. Returns StepResult."""  
        if self._dojo is None or self._current_state is None:  
            raise RuntimeError("No theorem loaded.")  
        if self._completed:  
            raise RuntimeError("Proof already complete.")  
  
        try:  
            result = self._dojo.run_tac(self._current_state, tactic)  
        except Exception as exc:  
            logger.warning("Tactic '%s' raised: %s", tactic, exc)  
            return StepResult(  
                success=False, state_text="", completed=False,  
                error_message=str(exc), tactic=tactic,  
            )  
  
        if isinstance(result, ProofFinished):  
            self._completed = True  
            return StepResult(  
                success=True, state_text="", completed=True,  
                error_message="", tactic=tactic,  
            )  
  
        if isinstance(result, TacticState):  
            self._current_state = result  
            return StepResult(  
                success=True, state_text=str(result.pp), completed=False,  
                error_message="", tactic=tactic,  
            )  
  
        # LeanError or other failure  
        error_msg = str(result) if isinstance(result, LeanError) else repr(result)  
        return StepResult(  
            success=False, state_text="", completed=False,  
            error_message=error_msg, tactic=tactic,  
        )  
  
    def close(self) -> None:  
        if self._dojo_context is not None:  
            try:  
                self._dojo_context.__exit__(None, None, None)  
            except Exception as exc:  
                logger.warning("Error closing Dojo: %s", exc)  
        self._dojo = None  
        self._current_state = None  
        self._theorem_info = None  
        self._completed = False  
        self._dojo_context = None  
  
    def __enter__(self):  
        return self  
  
    def __exit__(self, *args):  
        self.close()  
  
  
def theorem_info_from_dataset_row(row: dict) -> TheoremInfo:  
    """  
    Build TheoremInfo from a raw HuggingFace dataset row.  
  
    The LeanDojo benchmark rows have: url, commit, file_path, full_name.  
    Verify these field names match your dataset version by running  
    inspect_dataset.py first.  
    """  
    return TheoremInfo(  
        repo_url=row["url"],  
        commit=row["commit"],  
        file_path=row["file_path"],  
        full_name=row["full_name"],  
    )