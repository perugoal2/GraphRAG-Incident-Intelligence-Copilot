"""
retrieve.py — Vector, graph, and hybrid retrieval paths.

Hybrid merge design:
  - Graph candidates are scored by proximity: 1 / (hops + 1)
    (1-hop dependency = score 0.5, 2-hop = 0.33, etc.)
  - Each vector-similar incident boosts its root-cause candidate's score
    by the cosine similarity (0–1), but ONLY if that candidate is already
    in the graph candidate set.
  - "Structurally plausible AND historically precedented" ranks highest.
"""

from copilot.embed import embed
from copilot.graph import run


# ---------------------------------------------------------------------------
# Vector path
# ---------------------------------------------------------------------------

def vector_search(symptom: str, k: int = 5) -> list[dict]:
    """Find past incidents whose symptoms are semantically similar to *symptom*."""
    return run(
        """
        CALL db.index.vector.queryNodes('incident_embeddings', $k, $vec)
        YIELD node, score
        MATCH (node)-[:CAUSED_BY]->(rc:Service)
        RETURN node.id   AS id,
               node.symptom AS symptom,
               rc.name      AS root_cause,
               score
        ORDER BY score DESC
        """,
        k=k,
        vec=embed(symptom),
    )


# ---------------------------------------------------------------------------
# Graph path
# ---------------------------------------------------------------------------

def root_cause_candidates(svc: str, max_hops: int = 3) -> list[dict]:
    """
    Return services reachable by following DEPENDS_ON outward from *svc*.
    These are the structurally plausible root-cause candidates.
    max_hops=3 keeps traversal bounded on larger topologies.
    """
    return run(
        f"""
        MATCH (a:Service {{name: $svc}})
        MATCH path = (a)-[:DEPENDS_ON*1..{int(max_hops)}]->(cand:Service)
        RETURN DISTINCT cand.name AS candidate, min(length(path)) AS hops
        ORDER BY hops
        """,
        svc=svc,
    )


def blast_radius(root: str, max_hops: int = 3) -> list[dict]:
    """
    Return services that depend (directly or transitively) on *root*.
    Arrow direction is reversed vs. root_cause_candidates — inward toward root.
    """
    return run(
        f"""
        MATCH (r:Service {{name: $root}})<-[:DEPENDS_ON*1..{int(max_hops)}]-(aff:Service)
        RETURN DISTINCT aff.name AS affected
        """,
        root=root,
    )


# ---------------------------------------------------------------------------
# Hybrid merge — the core retrieval signal
# ---------------------------------------------------------------------------

def hybrid(symptom: str, affected_svc: str, k: int = 5) -> dict:
    """
    Combine graph proximity scores with vector similarity evidence.

    Returns:
      {
        ranked_candidates: [(name, score), ...],  sorted best-first
        similar_incidents: [{id, symptom, root_cause, score}, ...],
        blast_radius:      [{affected: name}, ...] for the top candidate
      }
    """
    similar = vector_search(symptom, k)
    candidates_raw = root_cause_candidates(affected_svc)

    # Base score: graph proximity (closer dependency = higher score)
    scores: dict[str, float] = {
        c["candidate"]: 1.0 / (c["hops"] + 1) for c in candidates_raw
    }

    # Boost: add vector similarity for candidates that appear in both signals
    for inc in similar:
        rc = inc["root_cause"]
        if rc in scores:
            scores[rc] += inc["score"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = ranked[0][0] if ranked else None

    return {
        "ranked_candidates": ranked,
        "similar_incidents": similar,
        "blast_radius": blast_radius(top) if top else [],
    }
