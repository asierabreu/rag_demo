"""
Prompt templates for the ESA Ground Segment RAG system.
All prompts are domain-tuned for spacecraft ground segment engineering.
"""
from __future__ import annotations

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert systems engineering assistant specialised in ESA \
(European Space Agency) spacecraft ground segment systems. You have deep expertise in:

• Mission Control Systems (MCS) — SCOS-2000, NCTRS, MOF
• Mission Planning Systems (MPS) — scheduling, timeline generation, conflict resolution
• Ground Station Interface Systems (GSIS) — SLE, antenna scheduling, RF chain management
• Telemetry & Telecommand (TM/TC) processing and CCSDS packet structures
• Flight Dynamics systems — orbit determination, manoeuvre planning, delta-V budgets
• ECSS standards (SW, SE, PA families) — tailoring, compliance, verification matrices
• ESA missions: Gaia, PLATO, CHEOPS, XMM-Newton, Mars Express, BepiColombo, Euclid, \
JUICE, and others

────────────────────────────────────────────
RULES:
1. Answer STRICTLY from the documentation context provided. Do not invent facts.
2. If the answer is not in the context, say so explicitly and suggest alternatives.
3. Always cite your sources using the reference numbers [1], [2], etc.
4. Use precise engineering terminology appropriate for senior systems engineers.
5. Be concise but complete. Prefer bullet points for multi-step answers.
6. When two sources conflict, flag the discrepancy explicitly.
────────────────────────────────────────────"""

# ── RAG prompt (context + history + query) ────────────────────────────────

RAG_PROMPT_TEMPLATE = """\
RETRIEVED DOCUMENTATION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENGINEER QUERY: {query}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide a technically precise answer based only on the documentation context above.
Cite sources with [n] notation. Flag any gaps or conflicts in the documentation."""

# ── No-context fallback ────────────────────────────────────────────────────

NO_CONTEXT_TEMPLATE = """\
The query "{query}" did not match any documentation in the current corpus \
(mission filter: {mission_filter}).

Possible reasons:
  1. The relevant documentation has not been ingested yet.
  2. The query uses terminology different from the document vocabulary — try synonyms.
  3. The selected mission filter excludes the relevant documents.

Suggestions:
  • Upload the relevant ICD, SRS, or technical note via the Documents panel.
  • Try rephrasing with alternative terms (e.g. "uplink" vs "telecommand", \
"MCS" vs "mission control").
  • Remove the mission filter to search across all missions."""

# ── Builder functions ──────────────────────────────────────────────────────

def build_rag_prompt(
    query: str,
    context: str,
    history: str = "",
) -> str:
    """Assemble the full user message for the RAG turn."""
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        history=history if history.strip() else "— no prior conversation —",
        query=query,
    )


def build_no_context_response(
    query: str,
    mission_filter: str | None = None,
) -> str:
    """Return a helpful message when retrieval returns nothing."""
    return NO_CONTEXT_TEMPLATE.format(
        query=query,
        mission_filter=mission_filter or "none (all missions searched)",
    )
