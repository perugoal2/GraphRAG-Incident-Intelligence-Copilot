"""
eval.py — Hit-rate, MRR, and ablation study.

Ablation design:
  - vector-only: candidates are the unique root-causes from the k most similar
                 past incidents, in similarity order.
  - graph-only:  candidates are services reachable via DEPENDS_ON from the
                 affected service, in hop-distance order.
  - hybrid:      candidates from the hybrid merge in composite-score order.

The delta between modes is the headline result: hybrid should outperform
either signal alone, proving the GraphRAG design earns its complexity.
"""

from copilot.graph import build_testset as _graph_build_testset
from copilot.retrieve import hybrid, root_cause_candidates, vector_search


def build_testset() -> list[dict]:
    """Return holdout incidents: [{id, symptom, affected, true_cause}]."""
    return _graph_build_testset()


def candidates_for(symptom: str, affected_svc: str, mode: str, k: int = 5) -> list[str]:
    if mode == "vector":
        seen: set[str] = set()
        ordered: list[str] = []
        for inc in vector_search(symptom, k):
            rc = inc["root_cause"]
            if rc not in seen:
                seen.add(rc)
                ordered.append(rc)
        return ordered

    if mode == "graph":
        return [c["candidate"] for c in root_cause_candidates(affected_svc)]

    # hybrid
    result = hybrid(symptom, affected_svc, k)
    return [name for name, _ in result["ranked_candidates"]]


def evaluate(testset: list[dict], mode: str, n: int = 3) -> dict:
    """
    Compute hit-rate@n and MRR for *mode* over *testset*.

    hit-rate@n: fraction of incidents where the true cause is in top-n candidates.
    MRR:        mean of 1/rank for each incident where the true cause appears
                anywhere in the candidates (rewards ranking it higher).
    """
    if not testset:
        return {f"hit_rate@{n}": 0.0, "mrr": 0.0}

    hits = 0
    reciprocal_rank_sum = 0.0

    for inc in testset:
        cands = candidates_for(inc["symptom"], inc["affected"], mode)
        true_cause = inc["true_cause"]

        if true_cause in cands[:n]:
            hits += 1

        if true_cause in cands:
            reciprocal_rank_sum += 1.0 / (cands.index(true_cause) + 1)

    total = len(testset)
    return {
        f"hit_rate@{n}": round(hits / total, 3),
        "mrr": round(reciprocal_rank_sum / total, 3),
    }


def ablation(testset: list[dict], n: int = 3) -> dict[str, dict]:
    """Run evaluate() for all three modes. Returns {mode: metrics}."""
    return {mode: evaluate(testset, mode, n) for mode in ("vector", "graph", "hybrid")}
