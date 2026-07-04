"""
chaos.py — Phase 2: Chaos-driven labelled incident generation.

Prerequisites (run once before using this module):
  1. Kubernetes cluster with Online Boutique deployed:
       kind create cluster --name incident-lab
       kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml

  2. Chaos Mesh installed (free, k8s-native):
       curl -sSL https://mirrors.chaos-mesh.org/v2.7.0/install.sh | bash

  3. Replace the synthetic topology with the real cluster graph by running
     load_topology_from_cluster() (heuristic) or by encoding the real
     service dependency edges in graph.py's DEPS constant.

Why this generates ground-truth labels:
  You inject a fault into a *known* service, observe the downstream symptoms,
  and record an Incident whose CAUSED_BY = the service you broke. The true
  root cause is known by construction — this is what makes rigorous evaluation
  possible without manual postmortem labelling.
"""

import os
import subprocess
import tempfile
import time
import uuid

from copilot.graph import run


# ---------------------------------------------------------------------------
# Core: write a chaos-labelled incident into the graph
# ---------------------------------------------------------------------------

def record_incident(
    inc_id: str,
    targeted_service: str,
    symptom_text: str,
    affected_services: list[str],
) -> None:
    """
    Persist a chaos-labelled incident. CAUSED_BY = the service you faulted.
    Sets synthetic=false to distinguish these from seeded incidents.
    """
    run(
        """
        MERGE (c:Service {name: $cause})
        CREATE (i:Incident {id: $id, symptom: $sym, synthetic: false, holdout: false})
        MERGE (i)-[:CAUSED_BY]->(c)
        WITH i
        UNWIND $affected AS a
        MERGE (s:Service {name: a})
        MERGE (i)-[:AFFECTED]->(s)
        """,
        id=inc_id,
        cause=targeted_service,
        sym=symptom_text,
        affected=affected_services,
    )


# ---------------------------------------------------------------------------
# Chaos Mesh manifest templates
# ---------------------------------------------------------------------------

_POD_CHAOS = """\
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: {name}
  namespace: default
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces: [default]
    labelSelectors:
      app: {service}
  duration: "30s"
"""

_NETWORK_CHAOS = """\
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: {name}
  namespace: default
spec:
  action: delay
  mode: all
  selector:
    namespaces: [default]
    labelSelectors:
      app: {service}
  delay:
    latency: "500ms"
    correlation: "100"
    jitter: "50ms"
  duration: "60s"
"""


def _apply_manifest(manifest_text: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(manifest_text)
        tmp = f.name
    try:
        subprocess.run(["kubectl", "apply", "-f", tmp], check=True)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Inject + observe + record (main Phase 2 entry point)
# ---------------------------------------------------------------------------

def inject_and_record(
    service: str,
    experiment_id: str | None = None,
    chaos_type: str = "pod-kill",
    observe_seconds: int = 90,
) -> str:
    """
    Inject a fault into *service*, wait for symptoms to propagate, then
    record a labelled Incident in the graph.

    Returns the incident ID written to Neo4j.

    NOTE: affected-service capture is graph-inferred (simpler v1).
    A production version would scrape Prometheus error rates during the
    observation window and record only services that actually degraded.
    """
    inc_id = experiment_id or f"CHAOS-{uuid.uuid4().hex[:8].upper()}"
    exp_name = f"incident-lab-{inc_id.lower()}"

    template = _POD_CHAOS if chaos_type == "pod-kill" else _NETWORK_CHAOS
    manifest = template.format(name=exp_name, service=service)

    print(f"[chaos] Injecting {chaos_type} into '{service}' (id={inc_id}) ...")
    _apply_manifest(manifest)

    print(f"[chaos] Observing for {observe_seconds}s ...")
    time.sleep(observe_seconds)

    # Infer affected services from the graph dependency structure.
    affected_rows = run(
        """
        MATCH (r:Service {name: $root})<-[:DEPENDS_ON*1..3]-(aff:Service)
        RETURN DISTINCT aff.name AS name
        """,
        root=service,
    )
    affected = [r["name"] for r in affected_rows] or [service]
    symptom = (
        f"{service} pod killed; downstream services reporting elevated error rates"
        if chaos_type == "pod-kill"
        else f"{service} experiencing 500ms network latency; dependent services timing out"
    )

    record_incident(inc_id, service, symptom, affected)

    # Embed the new incident so it's searchable immediately
    from copilot.embed import embed
    run(
        "MATCH (i:Incident {id: $id}) SET i.embedding = $v",
        id=inc_id,
        v=embed(symptom),
    )

    print(f"[chaos] Incident {inc_id} recorded. Affected services: {affected}")
    return inc_id


# ---------------------------------------------------------------------------
# Optional: infer topology from a live cluster
# ---------------------------------------------------------------------------

def load_topology_from_cluster() -> None:
    """
    Discover service names from the running cluster.
    Automated dependency inference requires Istio/Linkerd service-mesh telemetry.
    Without a mesh, encode DEPS in graph.py manually after reviewing the app.
    """
    result = subprocess.run(
        ["kubectl", "get", "pods", "-o", "jsonpath={.items[*].metadata.labels.app}"],
        capture_output=True,
        text=True,
        check=True,
    )
    service_names = sorted(set(result.stdout.strip().split()))
    print(f"Discovered {len(service_names)} services: {service_names}")
    print(
        "Next step: encode DEPENDS_ON edges in graph.py DEPS constant,\n"
        "or export from Istio: kubectl get serviceentries -o json"
    )
