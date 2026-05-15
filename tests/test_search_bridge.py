"""  
Unit tests for the state-to-graph bridge used in search.  
No LeanDojo required.  
"""  
  
from __future__ import annotations  
  
from atp_lean_gnn.graph import proof_state_to_dag  
from atp_lean_gnn.pyg import build_vocab, dag_to_pyg  
from atp_lean_gnn.state import parse_state  
  
  
def test_live_state_roundtrip():  
    """Simulate converting a live proof state through the pipeline."""  
    # This is what LeanDojo's state.pp would return  
    state_text = "n : ℕ\n⊢ 0 + n = n"  
  
    parsed = parse_state(state_text)  
    assert parsed.goal == "0 + n = n"  
    assert len(parsed.hypotheses) == 1  
    assert parsed.hypotheses[0].name == "n"  
  
    dag = proof_state_to_dag(parsed)  
    assert dag.num_nodes > 0  
  
    vocab = build_vocab([dag])  
    data = dag_to_pyg(dag, vocab)  
  
    assert data.x.dim() == 1  
    assert data.edge_index.dim() == 2  
    assert data.num_nodes == dag.num_nodes  
  
  
def test_multiple_hypotheses():  
    state_text = "a : ℕ\nb : ℕ\nh : a = b\n⊢ a + 1 = b + 1"  
  
    parsed = parse_state(state_text)  
    assert len(parsed.hypotheses) == 3  
    assert parsed.goal == "a + 1 = b + 1"  
  
    dag = proof_state_to_dag(parsed)  
    vocab = build_vocab([dag])  
    data = dag_to_pyg(dag, vocab)  
    assert data.num_nodes > 0