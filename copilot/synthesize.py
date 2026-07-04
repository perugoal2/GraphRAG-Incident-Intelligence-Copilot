"""
synthesize.py — LLM synthesis of grounded incident diagnoses.

Grounding strategy: the prompt passes ranked candidates, similar incidents,
and blast radius explicitly and instructs the model to cite only those IDs
and names. A root cause outside the candidate set is a detectable hallucination.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from copilot.retrieve import hybrid

load_dotenv()

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def diagnose(symptom: str, affected_svc: str, evidence: dict | None = None) -> str:
    """
    Produce a grounded diagnosis for a new incident.

    Pass pre-computed *evidence* (from hybrid()) to avoid a second retrieval
    round-trip when the caller already has it (e.g. the API endpoint).
    """
    ev = evidence if evidence is not None else hybrid(symptom, affected_svc)

    prompt = f"""A new incident: "{symptom}" affecting service "{affected_svc}".

Ranked root-cause candidates (name, composite-score):
{ev['ranked_candidates']}

Similar past incidents (most semantically similar first):
{json.dumps(ev['similar_incidents'], default=str)}

Predicted blast radius if the top candidate is the root:
{ev['blast_radius']}

Respond with:
1. The single most likely root cause, with concise reasoning.
2. The services most likely to be affected.
3. The most relevant past incident — cite its ID.

Rules:
- Ground every claim in the data above.
- Only name services that appear in the ranked candidates or blast radius.
- Only cite incident IDs that appear in the similar past incidents list.
- If the top candidate has a much higher score than the rest, say so.
- Be concise (3–6 sentences total)."""

    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
