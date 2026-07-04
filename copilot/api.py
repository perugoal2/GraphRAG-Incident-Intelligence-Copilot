"""
api.py — FastAPI endpoints for the Incident Intelligence Copilot.

Start:
  uvicorn copilot.api:app --reload

Endpoints:
  POST /diagnose           → grounded diagnosis + ranked candidates + blast radius
  GET  /subgraph/{service} → node/edge graph for visualization
  GET  /health             → liveness check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from copilot.graph import run
from copilot.retrieve import hybrid
from copilot.synthesize import diagnose as _diagnose

app = FastAPI(
    title="Incident Intelligence Copilot",
    description="GraphRAG-powered incident root-cause analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /diagnose
# ---------------------------------------------------------------------------

class DiagnoseRequest(BaseModel):
    symptom: str
    service: str


class Candidate(BaseModel):
    name: str
    score: float


class DiagnoseResponse(BaseModel):
    diagnosis: str
    ranked_candidates: list[Candidate]
    blast_radius: list[dict]
    similar_incidents: list[dict]


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_incident(req: DiagnoseRequest) -> DiagnoseResponse:
    """
    Retrieve evidence via hybrid GraphRAG and synthesize a grounded diagnosis.
    Calls hybrid() once; passes the evidence into both the response payload
    and the LLM synthesis to avoid a redundant retrieval round-trip.
    """
    if not req.symptom.strip():
        raise HTTPException(status_code=400, detail="symptom must not be empty")
    if not req.service.strip():
        raise HTTPException(status_code=400, detail="service must not be empty")

    evidence = hybrid(req.symptom, req.service)
    diagnosis_text = _diagnose(req.symptom, req.service, evidence=evidence)

    return DiagnoseResponse(
        diagnosis=diagnosis_text,
        ranked_candidates=[
            Candidate(name=name, score=round(score, 4))
            for name, score in evidence["ranked_candidates"]
        ],
        blast_radius=evidence["blast_radius"],
        similar_incidents=evidence["similar_incidents"],
    )


# ---------------------------------------------------------------------------
# /subgraph/{service}
# ---------------------------------------------------------------------------

class SubgraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@app.get("/subgraph/{service}", response_model=SubgraphResponse)
def get_subgraph(service: str, max_hops: int = 2) -> SubgraphResponse:
    """
    Return the dependency subgraph centred on *service* (up to max_hops deep)
    plus any incidents that affected it — useful for graph visualisation.
    """
    if max_hops < 1 or max_hops > 5:
        raise HTTPException(status_code=400, detail="max_hops must be between 1 and 5")

    service_nodes = run(
        f"""
        MATCH (s:Service {{name: $svc}})
        OPTIONAL MATCH (s)-[:DEPENDS_ON*1..{int(max_hops)}]->(dep:Service)
        OPTIONAL MATCH (up:Service)-[:DEPENDS_ON*1..{int(max_hops)}]->(s)
        WITH collect(DISTINCT s) + collect(DISTINCT dep) + collect(DISTINCT up) AS all_nodes
        UNWIND all_nodes AS n
        RETURN DISTINCT n.name AS id, 'Service' AS label
        """,
        svc=service,
    )

    incident_nodes = run(
        """
        MATCH (inc:Incident)-[:AFFECTED]->(s:Service {name: $svc})
        RETURN inc.id AS id, 'Incident' AS label, inc.symptom AS symptom
        LIMIT 10
        """,
        svc=service,
    )

    edges = run(
        f"""
        MATCH (a:Service)-[:DEPENDS_ON]->(b:Service)
        WHERE (a.name = $svc OR b.name = $svc)
          OR  EXISTS {{
                MATCH (a)-[:DEPENDS_ON*1..{int(max_hops)}]-(s:Service {{name: $svc}})
              }}
        RETURN DISTINCT a.name AS source, b.name AS target, 'DEPENDS_ON' AS type
        """,
        svc=service,
    )

    incident_edges = run(
        """
        MATCH (inc:Incident)-[:AFFECTED]->(s:Service {name: $svc})
        RETURN inc.id AS source, s.name AS target, 'AFFECTED' AS type
        LIMIT 10
        """,
        svc=service,
    )

    return SubgraphResponse(
        nodes=service_nodes + incident_nodes,
        edges=edges + incident_edges,
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
