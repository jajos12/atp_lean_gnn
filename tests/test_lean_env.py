"""  
Smoke tests for the LeanDojo environment wrapper.  
Skipped if lean-dojo is not installed.  
"""  
  
from __future__ import annotations  
import re 
import pytest  
  
try:  
    import lean_dojo  
    HAS_LEAN_DOJO = True  
except ImportError:  
    HAS_LEAN_DOJO = False  
  
requires_leandojo = pytest.mark.skipif(not HAS_LEAN_DOJO, reason="lean-dojo not installed")  
  
  
@requires_leandojo  
def test_load_and_apply():  
    """Load a theorem, apply a tactic, verify we get a result."""  
    from datasets import load_dataset  
    from atp_lean_gnn.lean_env import LeanEnvironment, theorem_info_from_dataset_row  
  
    # Grab one real row from the dataset  
    ds = load_dataset(  
        "cat-searcher/leandojo-benchmark-4-random",  
        split="test", streaming=True,  
    )  
    row = next(iter(ds))  
    info = theorem_info_from_dataset_row(row)  
  
    with LeanEnvironment() as env:  
        state = env.load_theorem(info)  
        assert isinstance(state, str)  
        assert len(state) > 0  
        assert not env.is_complete()  
  
        # Strip <a>...</a> tags and apply the gold tactic  
        tactic = re.sub(r"</?a[^>]*>", "", row["tactic"])  
        result = env.apply_tactic(tactic) 
        # It should either succeed or give a clean error  
        assert result.success or len(result.error_message) > 0  
  
  
@requires_leandojo  
def test_invalid_tactic():  
    from datasets import load_dataset  
    from atp_lean_gnn.lean_env import LeanEnvironment, theorem_info_from_dataset_row  
  
    ds = load_dataset(  
        "cat-searcher/leandojo-benchmark-4-random",  
        split="test", streaming=True,  
    )  
    row = next(iter(ds))  
    info = theorem_info_from_dataset_row(row)  
  
    with LeanEnvironment() as env:  
        env.load_theorem(info)  
        result = env.apply_tactic("not_a_real_tactic_xyz")  
        assert not result.success  
        assert len(result.error_message) > 0