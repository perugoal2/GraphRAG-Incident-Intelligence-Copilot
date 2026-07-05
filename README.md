# GraphRAG Incident-Intelligence Copilot

An on-call assistant that combines a **knowledge graph** (Neo4j) and **vector search** (sentence-transformers) to diagnose microservice incidents — finding the root cause and blast radius faster than either signal alone.

> The **graph** holds *structure* — who depends on whom.  
> The **vector index** holds *meaning* — what past incidents looked like.  
> **Hybrid retrieval** combines the two.  
> The **LLM** writes a grounded diagnosis.  
> **Chaos engineering** generates labelled data to prove it works.

---

## Architecture

```
New Incident Symptom
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  Hybrid Retrieval                     │
│                                                       │
│  ┌─────────────────┐      ┌──────────────────────┐   │
│  │  Vector Search  │      │    Graph Traversal   │   │
│  │                 │      │                      │   │
│  │  Embed symptom  │      │  DEPENDS_ON outward  │   │
│  │  → cosine sim   │      │  from affected svc   │   │
│  │  → past IDs +   │      │  → candidates scored │   │
│  │    root causes  │      │    by 1/(hops+1)     │   │
│  └────────┬────────┘      └──────────┬───────────┘   │
│           │    Cross-signal boost    │               │
│           └──────────────┬───────────┘               │
│                          ▼                            │
│              Ranked Root-Cause Candidates             │
│              + Blast Radius (inward traversal)        │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
                    LLM Synthesis
              (claude-sonnet-4-6, grounded,
               cites incident IDs & service names)
                           │
                           ▼
                 Grounded Diagnosis
```

**Neo4j stores both the graph topology and the embedding vectors on the same nodes** — one query can do similarity search and relationship traversal simultaneously. That's the GraphRAG architectural advantage.

### Edge direction

`A -[:DEPENDS_ON]-> B` means A needs B; B is upstream of A.

- **Root-cause candidates**: follow `DEPENDS_ON` *outward* from the affected service (find what it depends on).
- **Blast radius**: follow `DEPENDS_ON` *inward* toward the suspected root (find everything that depends on it).

---

## Project structure

```
incident-copilot/
├── copilot/
│   ├── graph.py       Neo4j driver, topology + incident loaders, vector index
│   ├── embed.py       Sentence-transformer embedding (all-MiniLM-L6-v2, 384-dim)
│   ├── retrieve.py    Vector path, graph path, hybrid merge
│   ├── synthesize.py  LLM synthesis (Anthropic)
│   ├── cli.py         init / ask / ablate commands
│   ├── chaos.py       Phase 2: fault injection → labelled incidents
│   ├── eval.py        Phase 2: hit-rate, MRR, ablation
│   └── api.py         FastAPI: POST /diagnose, GET /subgraph/{service}
├── data/
├── docker-compose.yml  Neo4j 5
└── .env                ANTHROPIC_API_KEY, NEO4J_URI, NEO4J_PASSWORD
```

---

## Quick start

### Prerequisites

- Docker Desktop
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com)

### 1 — Start Neo4j

```bash
docker compose up -d
# Browser UI: http://localhost:7474  (neo4j / password123)
```

### 2 — Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Configure credentials

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...      # your key from console.anthropic.com
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password123
```

### 4 — Bootstrap the database

```bash
python -m copilot.cli init
# Loads topology → seeds 25 incidents (20 training / 5 holdout) → creates vector index → embeds
```

### 5 — Ask a question

```bash
python -m copilot.cli ask --symptom "checkout throwing 503s, latency spiked" --service checkout
```

Example output:
```
--- Diagnosis ---

Most likely root cause: shipping (composite score 2.15, corroborated by INC-016).
Shipping is a direct dependency of checkout and has a history of connection-pool
exhaustion under load. If confirmed, expected blast radius: checkout, frontend.
Most relevant past incident: INC-016 — "connection pool exhausted in shipping".
```

### 6 — Run the ablation study

```bash
# On synthetic holdout incidents
python -m copilot.cli ablate

# On real chaos-labelled incidents (Phase 2, requires kind cluster)
python -m copilot.cli ablate --chaos
```

### 7 — Start the API server

```bash
uvicorn copilot.api:app --reload
# Docs: http://localhost:8000/docs
```

`POST /diagnose`:
```json
{ "symptom": "checkout throwing 503s", "service": "checkout" }
```

`GET /subgraph/checkout?max_hops=2` — returns nodes + edges for graph visualisation.

---

## Phase 2 — Chaos-driven labelled data

Phase 2 generates incidents with known root causes by injecting real faults and observing the downstream symptoms.

### Prerequisites

```bash
# Install kind
curl -Lo kind.exe https://kind.sigs.k8s.io/dl/v0.29.0/kind-windows-amd64
# (move to a directory on PATH)

# Create cluster
kind create cluster --name incident-lab

# Deploy Online Boutique
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml

# Install Chaos Mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org && helm repo update
kubectl create namespace chaos-testing
helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace chaos-testing \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
  --version 2.7.1
```

### Inject and record

```python
from copilot.chaos import inject_and_record

inject_and_record("cart",     experiment_id="CHAOS-001", chaos_type="pod-kill")
inject_and_record("payment",  experiment_id="CHAOS-002", chaos_type="pod-kill")
inject_and_record("shipping", experiment_id="CHAOS-003", chaos_type="network-delay")
```

Use graph short-names: `cart`, `checkout`, `payment`, `shipping`, `redis`, `productcatalog`, `recommendation`, `ad`, `email`, `frontend`.

Each call:
1. Applies a Chaos Mesh manifest targeting the matching k8s pod
2. Observes for 35s (configurable via `observe_seconds`)
3. Writes an `Incident` node in Neo4j with `CAUSED_BY = targeted_service` and embedds it

---

## Ablation results

Evaluated on 9 chaos-injected incidents with known root causes (CHAOS-002 through CHAOS-010).

| Retrieval mode | Hit-Rate@3 | MRR   |
|----------------|------------|-------|
| vector-only    | 1.000      | 1.000 |
| graph-only     | 0.778      | 0.569 |
| **hybrid**     | **1.000**  | **0.944** |

**Key finding:** Graph-only drops to 0.778 hit-rate and 0.569 MRR — it finds candidates structurally but ranks the true root cause 2nd or 3rd in 2 of 9 cases. Hybrid matches vector on hit-rate while adding structural grounding (blast radius, candidate filtering) that pure vector search can't provide.

*Note: 9-incident test set is indicative. Run more chaos experiments across remaining services to strengthen the signal.*

---

## Design decisions

| Decision | Reasoning |
|---|---|
| GraphRAG over flat RAG | Structure answers questions meaning can't: root cause (outward traversal) and blast radius (inward traversal). Ablation confirms it: graph-only 0.778 vs hybrid 1.0 hit-rate. |
| `A DEPENDS_ON B` edge direction | B is upstream of A. Causes follow arrows outward; blast radius follows them inward. Getting this backwards inverts every retrieval result. |
| Vectors stored on graph nodes | One store, one query does similarity search and relationship traversal simultaneously — no ETL between a separate vector DB and a graph DB. |
| Hybrid merge scoring | Graph candidates are scored by `1/(hops+1)` (closer dependency = higher base score). Vector similarity boosts any candidate that also appears in semantically similar past incidents. "Structurally plausible and historically precedented" ranks highest. |
| Chaos for ground-truth labels | Injecting the fault means `CAUSED_BY` is known by construction — no manual postmortem labelling, no ambiguous multi-cause incidents to reconcile. |
| Grounded LLM prompt | The prompt passes ranked candidates, similar incident IDs, and blast radius explicitly and instructs the model to cite only those. A root cause outside `ranked_candidates` in the response is a detectable hallucination. |

---

## What I'd do at scale

- **Graph size:** Cap `max_hops` at 2–3 on a 500-service graph; add a `[:CRITICAL]` edge property to weight direct dependencies higher than transitive ones.
- **Embedding refresh:** Stream new incidents into a queue (Kafka/Pub-Sub); a worker calls `embed()` and updates the vector index incrementally — no full reindex.
- **Observability integration:** Replace the graph-inferred blast radius with live Prometheus error rates during the chaos observation window for higher-fidelity affected-service capture.
- **Multi-root causes:** The current model assumes one root per incident; extend with `CAUSED_BY` confidence scores and allow the LLM to reason about correlated failures.
- **Chaos coverage:** Automate a nightly suite that cycles through all services, builds a labelled corpus, and re-runs the ablation — treating it as a regression test for retrieval quality.
