"""
graph.py — Neo4j connection, schema, topology and incident loaders.

Edge-direction convention (critical):
  A -[:DEPENDS_ON]-> B  means  A needs B; B is *upstream* of A.
  - Root-cause candidates: follow DEPENDS_ON *outward* from affected service.
  - Blast radius:          follow DEPENDS_ON *inward* toward a suspected root.
"""

import os
import random

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=("neo4j", os.getenv("NEO4J_PASSWORD", "password123")),
)


def run(cypher: str, **params) -> list:
    with _driver.session() as session:
        return [r.data() for r in session.run(cypher, **params)]


# ---------------------------------------------------------------------------
# Online Boutique topology (hardcoded for Phase 1 synthetic data)
# ---------------------------------------------------------------------------

DEPS: list[tuple[str, str]] = [
    # (service, depends_on)  →  service needs depends_on
    ("frontend", "productcatalog"),
    ("frontend", "cart"),
    ("frontend", "checkout"),
    ("frontend", "recommendation"),
    ("frontend", "ad"),
    ("checkout", "payment"),
    ("checkout", "shipping"),
    ("checkout", "email"),
    ("checkout", "cart"),
    ("cart", "redis"),
    ("recommendation", "productcatalog"),
]

_SERVICES: list[str] = list({s for s, _ in DEPS} | {d for _, d in DEPS})

_TEMPLATES: list[tuple[str, str]] = [
    ("{svc} returning 503s, latency spiked",         "connection pool exhausted in {cause}"),
    ("checkout failing intermittently",               "{cause} timing out under load"),
    ("blank product pages reported",                  "{cause} returned malformed responses"),
    ("{svc} health-check failures increasing",        "{cause} OOMKilled repeatedly"),
    ("{svc} response time p99 > 5s",                  "{cause} disk I/O saturated"),
    ("{svc} error rate jumped to 30%",                "{cause} certificate expired"),
    ("users unable to add items to cart",             "{cause} connection refused"),
    ("{svc} logs showing repeated timeouts",          "{cause} memory leak under sustained load"),
]


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def load_topology() -> None:
    run("MATCH (n) DETACH DELETE n")
    for svc, dep in DEPS:
        run(
            """
            MERGE (a:Service {name: $svc})
            MERGE (b:Service {name: $dep})
            MERGE (a)-[:DEPENDS_ON]->(b)
            """,
            svc=svc,
            dep=dep,
        )
    print(f"Topology loaded: {len(_SERVICES)} services, {len(DEPS)} dependency edges.")


# ---------------------------------------------------------------------------
# Incidents + postmortems
# ---------------------------------------------------------------------------

def seed_incidents(n: int = 25, holdout_frac: float = 0.2) -> None:
    """
    Create n synthetic incidents with known CAUSED_BY labels.
    The last (holdout_frac * n) incidents are marked holdout=true and excluded
    from embeddings — they form the evaluation test set.
    """
    holdout_start = int(n * (1 - holdout_frac))

    for i in range(n):
        svc = random.choice(_SERVICES)
        deps_of_svc = [dep for src, dep in DEPS if src == svc]
        cause = random.choice(deps_of_svc) if deps_of_svc else svc

        sym_tpl, root_tpl = random.choice(_TEMPLATES)
        sym = sym_tpl.format(svc=svc)
        txt = root_tpl.format(cause=cause)
        holdout = i >= holdout_start

        run(
            """
            MERGE (s:Service {name: $svc})
            MERGE (c:Service {name: $cause})
            CREATE (inc:Incident {id: $id, symptom: $sym, holdout: $holdout})
            CREATE (p:Postmortem {id: $pid, text: $txt})
            MERGE (inc)-[:AFFECTED]->(s)
            MERGE (inc)-[:CAUSED_BY]->(c)
            MERGE (p)-[:DOCUMENTS]->(inc)
            """,
            svc=svc,
            cause=cause,
            id=f"INC-{i:03d}",
            pid=f"PM-{i:03d}",
            sym=sym,
            txt=txt,
            holdout=holdout,
        )

    n_holdout = n - holdout_start
    print(f"Seeded {n} incidents ({n_holdout} held out for evaluation, {n - n_holdout} in training index).")


# ---------------------------------------------------------------------------
# Vector index + embeddings
# ---------------------------------------------------------------------------

def create_vector_index() -> None:
    run(
        """
        CREATE VECTOR INDEX incident_embeddings IF NOT EXISTS
        FOR (i:Incident) ON (i.embedding)
        OPTIONS { indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }}
        """
    )
    print("Vector index 'incident_embeddings' created (or already exists).")


def embed_incidents() -> None:
    """
    Embed all non-holdout incidents that don't yet have an embedding.
    Holdout incidents are intentionally skipped to prevent evaluation leakage.
    """
    from copilot.embed import embed  # deferred to avoid import at module load

    rows = run(
        "MATCH (i:Incident) WHERE i.holdout = false AND i.embedding IS NULL "
        "RETURN i.id AS id, i.symptom AS sym"
    )
    for r in rows:
        run(
            "MATCH (i:Incident {id: $id}) SET i.embedding = $v",
            id=r["id"],
            v=embed(r["sym"]),
        )
    print(f"Embedded {len(rows)} incidents.")


# ---------------------------------------------------------------------------
# Evaluation testset
# ---------------------------------------------------------------------------

def build_testset() -> list[dict]:
    """Return holdout incidents as [{id, symptom, affected, true_cause}]."""
    return run(
        """
        MATCH (inc:Incident {holdout: true})-[:AFFECTED]->(s:Service),
              (inc)-[:CAUSED_BY]->(c:Service)
        RETURN inc.id AS id, inc.symptom AS symptom,
               s.name AS affected, c.name AS true_cause
        """
    )


def build_chaos_testset() -> list[dict]:
    """
    Return chaos-labelled (synthetic=false) incidents as [{id, symptom, affected, true_cause}].
    Each incident appears once — affected is the first downstream service found
    (any valid starting point yields the same root-cause candidates via graph traversal).
    CHAOS-001 is excluded: it used the raw k8s label before the name-mapping fix.
    """
    return run(
        """
        MATCH (inc:Incident {synthetic: false})-[:AFFECTED]->(s:Service),
              (inc)-[:CAUSED_BY]->(c:Service)
        WHERE inc.id <> 'CHAOS-001'
        WITH inc, s, c
        ORDER BY inc.id, s.name
        WITH inc, head(collect(s.name)) AS affected, c.name AS true_cause
        RETURN inc.id AS id, inc.symptom AS symptom,
               affected, true_cause
        ORDER BY inc.id
        """
    )
