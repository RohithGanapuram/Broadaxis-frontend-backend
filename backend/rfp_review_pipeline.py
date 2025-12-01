# backend/backend/rfp_review_pipeline.py
from typing import Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import hashlib
import time
import re
import asyncio


from sharepoint_api import download_package
from mcp_interface import run_mcp_query, REVIEW_PACKAGE_SYSTEM_PROMPT


ProgressCB = Callable[[str, float, Dict[str, Any]], None]

FINAL_VERDICT_SYSTEM_PROMPT = """
You are a senior public-sector RFP capture manager.

You are given:
- A short summary of an RFP package (agency, scope, requirements).
- The results of an eligibility check (which mandatory requirements look satisfied or missing).
- A very high-level view of the proposal documents we submitted.

Your job is to answer ONE question: CAN WE LIKELY WIN THIS RFP OR NOT?

Rules:
- Think like an evaluator who has seen many RFPs and proposals.
- Use your general knowledge of how RFPs are scored worldwide.
- Do NOT invent new documents; only reason from the summary you receive.
- If critical mandatory items or signature forms are missing, you should lean to NO_GO / UNLIKELY.

You MUST respond with a single JSON object ONLY, in this exact schema:

{
  "verdict": "LIKELY_WIN" | "COMPETITIVE" | "UNLIKELY" | "NO_GO",
  "confidence": 0.0–1.0,
  "reasons": [
    "short bullet reason 1",
    "short bullet reason 2",
    "short bullet reason 3"
  ]
}

Do not add any text before or after the JSON.
"""


@dataclass
class Doc:
    doc_id: str
    name: str
    type: str
    bytes: int
    text: str
    anchors: List[Dict[str, Any]]
    detected_role: str

RUBRIC_WEIGHTS = {
    "criteria_coverage": 0.50,
    "capability_fit": 0.25,
    "hygiene": 0.25,
}

LABELS = [
    (85, "Strong Favorite"),
    (70, "Likely Competitive"),
    (55, "Toss-up"),
    (40, "Weak"),
    (0,  "No-Go (fix blockers)"),
]

def _hash_package(file_records: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in sorted(file_records, key=lambda x: (x.get('id') or x.get('name'), x.get('last_modified'))):
        h.update(str(r.get('id') or r.get('name')).encode())
        h.update(str(r.get('size', 0)).encode())
        h.update(str(r.get('last_modified', '')).encode())
    return h.hexdigest()[:16]

def _emit(progress_cb: ProgressCB, message: str, pct: float, extra: Dict[str, Any] = None):
    if progress_cb:
        progress_cb(message, pct, extra or {})

def ingest_package(path: str, options: Dict[str, Any], progress_cb: ProgressCB) -> Tuple[List[Doc], Dict[str, Any]]:
    _emit(progress_cb, "Indexing package…", 0.05)
    pkg = download_package(path, include_subfolders=options.get("include_subfolders", True),
                           max_files=options.get("max_files", 200))
    files = pkg.get("files", [])
    package_hash = _hash_package(files)

    docs: List[Doc] = []

    def detect_role(name: str, text: str) -> str:
        lname = name.lower()
        if any(k in lname for k in ["evaluation", "scoring", "criteria"]):
            return "rfp_core"
        if any(k in lname for k in ["resume", "cv"]):
            return "resume"
        if any(k in lname for k in ["price", "cost", "fee", "pricing"]):
            return "pricing"
        if any(k in lname for k in ["form", "attachment", "att "]):
            return "form"
        if any(k in lname for k in ["narrative", "approach", "methodology"]):
            return "narrative"
        return "other"

    for i, f in enumerate(files):
        text = f.get("text", "")
        docs.append(Doc(
            doc_id=str(f.get("id") or f.get("name")),
            name=f.get("name", "unknown"),
            type=f.get("mime", "unknown"),
            bytes=int(f.get("size", 0)),
            text=text,
            anchors=[{"page": None, "section": None}],
            detected_role=detect_role(f.get("name", ""), text)
        ))
        if i % 10 == 0:
            _emit(progress_cb, "Indexing package…", 0.05 + 0.02 * i)

    return docs, {"hash": package_hash, "file_count": len(files)}

def extract_requirements(docs: List[Doc], progress_cb: ProgressCB) -> Dict[str, Any]:
    _emit(progress_cb, "Extracting requirements…", 0.20)
    all_text = "\n\n".join(d.text for d in docs if d.detected_role in ("rfp_core", "narrative", "other"))

    elig = []
    for m in re.finditer(r"(?im)\b(shall|must|required|mandatory)\b.*?\.", all_text):
        t = m.group(0)
        elig.append({
            "id": f"REQ-{len(elig)+1:03d}",
            "requirement": t[:500],
            "source_anchor": "RFP_Core (approx)",
            "required_evidence_type": None
        })

    criteria = [
        {"name": "Technical Approach", "weight": None, "signals": ["approach", "methodology", "architecture"]},
        {"name": "Staffing & Key Personnel", "weight": None, "signals": ["resume", "key personnel", "project manager"]},
        {"name": "Past Performance", "weight": None, "signals": ["past performance", "references", "similar projects"]},
    ]
    return {"eligibility": elig[:30], "criteria": criteria}

def kb_evidence_check(requirements: Dict[str, Any], progress_cb: ProgressCB, session_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _emit(progress_cb, "Checking BroadAxis evidence…", 0.45)
    eligibility_out = []
    for i, r in enumerate(requirements["eligibility"]):
        query = r["requirement"]

        # Call MCP using only the supported arguments
        result = asyncio.run(run_mcp_query(
            query=f"Find evidence for: {query}",
            enabled_tools=["Broadaxis_knowledge_search"],
            model=None,
            session_id=session_id,
            system_prompt=REVIEW_PACKAGE_SYSTEM_PROMPT,
        ))

        hits = result.get("tool_results", []) if isinstance(result, dict) else []
        best = hits[0] if hits else {}
        score = float(best.get("score", 0)) if best else 0.0
        status = "PASS" if score >= 0.82 else ("UNCLEAR" if score >= 0.70 else "FAIL")
        eligibility_out.append({
            **r,
            "evidence": {
                "status": status,
                "kb_doc_title": best.get("title") or best.get("source"),
                "snippet": (best.get("text") or "")[:400],
                "confidence": score
            }
        })
        if i % 5 == 0:
            _emit(progress_cb, "Checking BroadAxis evidence…", 0.45 + 0.02 * i)


    # MVP placeholders for criteria
    criteria_out = []
    for c in requirements["criteria"]:
        criteria_out.append({
            "criterion": c["name"],
            "coverage": 0.7,
            "strength": 0.6,
            "notes": ["MVP placeholder – inspect narratives and KB for real values"]
        })
    return eligibility_out, criteria_out


def _build_verdict_summary(path: str,
                           docs: List[Doc],
                           eligibility: List[Dict[str, Any]],
                           hygiene: List[Dict[str, Any]]) -> str:
    """Build a short text summary for the final 'can we win?' verdict call."""
    package_name = path.split("/")[-1]

    # Basic doc inventory (only names + rough roles)
    doc_lines = []
    for d in docs[:40]:  # cap to keep it small
        doc_lines.append(f"- {d.name} [{d.detected_role}]")

    # Eligibility summary
    elig_lines = []
    fail_count = 0
    for e in eligibility:
        ev = e.get("evidence", {})
        status = ev.get("status", "UNKNOWN")
        conf = ev.get("confidence", 0.0)
        if status == "FAIL":
            fail_count += 1
        elig_lines.append(
            f"- {e['id']}: {status} (conf={conf:.2f}) — {e['requirement'][:180]}"
        )

    # Hygiene summary
    hygiene_lines = []
    for h in hygiene:
        hygiene_lines.append(
            f"- {h['item']}: {h.get('status','UNKNOWN')} ({h.get('fix','').strip()})"
        )

    summary = [
    f"PACKAGE: {package_name}",
    f"PATH: {path}",
    "",
    "DOCUMENT INVENTORY (name [role]):",
    ]

    summary.extend(doc_lines if doc_lines else ["- (no docs found)"])

    summary += [
        "",
        f"ELIGIBILITY SUMMARY (total={len(eligibility)}, fails={fail_count}):",
    ]

    summary.extend(elig_lines if elig_lines else ["- (no eligibility requirements extracted)"])

    summary += [
        "",
        "HYGIENE FINDINGS:",
    ]

    summary.extend(hygiene_lines if hygiene_lines else ["- (no hygiene issues detected)"])

    return "\n".join(summary)


import json  # at top if not already

def model_win_verdict(path: str,
                      docs: List[Doc],
                      eligibility: List[Dict[str, Any]],
                      hygiene: List[Dict[str, Any]],
                      session_id: str,
                      progress_cb: ProgressCB) -> Dict[str, Any]:
    """
    Ask the LLM for a final 'can we win?' verdict, based on a summary of this package.
    Returns: {"verdict": str, "confidence": float, "reasons": [str,...]}
    """
    _emit(progress_cb, "Computing overall win verdict…", 0.9)

    # Hard gate: if any FAIL, strongly bias the model toward NO_GO, but still let it see the full picture.
    has_fail = any(e.get("evidence", {}).get("status") == "FAIL" for e in eligibility)

    summary_text = _build_verdict_summary(path, docs, eligibility, hygiene)

    user_query = f"""
Here is the analysis of an RFP package and our submission:

<ANALYSIS>
{summary_text}
</ANALYSIS>

Based on this analysis, use your experience with RFPs and proposals
to decide if we are likely to win this RFP or not.

Remember to respond ONLY with the JSON object as specified in the system prompt.
"""

    # No tools – this is pure reasoning.
    result = asyncio.run(run_mcp_query(
        query=user_query,
        enabled_tools=[],
        model=None,
        session_id=session_id,
        system_prompt=FINAL_VERDICT_SYSTEM_PROMPT,
    ))

    raw = result.get("response", "") if isinstance(result, dict) else str(result)

    # Extract JSON object from the response
    json_text = None
    try:
        # naive: find first '{' ... last '}'
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = raw[start:end+1]
        data = json.loads(json_text)
    except Exception:
        # Fallback: very defensive default
        data = {
            "verdict": "UNLIKELY" if has_fail else "COMPETITIVE",
            "confidence": 0.5,
            "reasons": [
                "Failed to parse model JSON response; using conservative default."
            ]
        }

    # Normalize fields
    verdict = (data.get("verdict") or "COMPETITIVE").upper()
    if verdict not in ("LIKELY_WIN", "COMPETITIVE", "UNLIKELY", "NO_GO"):
        verdict = "COMPETITIVE"

    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    reasons = data.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]

    return {
        "verdict": verdict,
        "confidence": conf,
        "reasons": reasons,
    }

def hygiene_checks(docs: List[Doc], progress_cb: ProgressCB) -> List[Dict[str, Any]]:
    _emit(progress_cb, "Running document hygiene checks…", 0.70)
    findings = []
    names = [d.name.lower() for d in docs]
    if not any("attachment b" in n for n in names):
        findings.append({"item": "Attachment B – Signature", "status": "MISSING", "fix": "Collect signature"})
    if not any("form 1295" in n for n in names):
        findings.append({"item": "Form 1295", "status": "MISSING", "fix": "Generate and include Form 1295"})
    return findings

def score_and_label(eligibility: List[Dict[str, Any]], criteria: List[Dict[str, Any]], hygiene: List[Dict[str, Any]]) -> Tuple[int, str, Dict[str, float]]:
    if any(e.get("evidence", {}).get("status") == "FAIL" for e in eligibility):
        base = 39
        label = next(lbl for thr, lbl in LABELS if base >= thr)
        return base, label, {k: 0.2 for k in RUBRIC_WEIGHTS}

    cov = sum(c.get("coverage", 0) * c.get("strength", 0) for c in criteria) / max(1, len(criteria))
    fit = 0.75  # MVP placeholder; replace with KB-derived fit
    hyg = 1.0 - min(1.0, len([f for f in hygiene if f.get("status") == "MISSING"]) * 0.2)

    subs = {"criteria_coverage": cov, "capability_fit": fit, "hygiene": hyg}
    total = sum(subs[k] * RUBRIC_WEIGHTS[k] for k in RUBRIC_WEIGHTS)
    rating = int(round(total * 100))
    for thr, lbl in LABELS:
        if rating >= thr:
            return rating, lbl, subs
    return rating, LABELS[-1][1], subs

def format_report(
    package_name: str,
    path: str,
    eligibility_status: str,
    rating: int,
    label: str,
    eligibility: List[Dict[str, Any]],
    criteria: List[Dict[str, Any]],
    hygiene: List[Dict[str, Any]],
    subs: Dict[str, Any],  # <-- changed to Any, because we now store model_verdict dict here
    cache_key: str,
) -> Dict[str, Any]:
    verdict_info = subs.get("model_verdict", {})
    v = verdict_info.get("verdict", "COMPETITIVE")
    conf = verdict_info.get("confidence", 0.5)
    reasons = verdict_info.get("reasons") or []

    md = [
        f"# Review Package — {package_name}",
        "",
        f"**Model Verdict:** **{v}** (confidence {conf:.2f} → {rating}/100)",
        f"**Eligibility:** **{eligibility_status}**",
        "",
        "### Why the model thinks this:",
    ]
    if reasons:
        md.extend([f"- {r}" for r in reasons])
    else:
        md.append("- (no reasons provided)")

    # If you want to keep the old scorecard, you can uncomment this block
    # but then you MUST ensure subs has those numeric keys.
    #
    # md += [
    #     "",
    #     "## Scorecard",
    #     f"- Criteria coverage: **{subs.get('criteria_coverage', 0.0):.2f}**",
    #     f"- Capability fit: **{subs.get('capability_fit', 0.0):.2f}**",
    #     f"- Hygiene: **{subs.get('hygiene', 0.0):.2f}**",
    # ]

    md += [
        "",
        "## Criteria Findings",
    ]
    for c in criteria:
        md.append(
            f"- **{c['criterion']}** — coverage {c.get('coverage', 0.0):.2f}, "
            f"strength {c.get('strength', 0.0):.2f}. "
            f"Notes: {', '.join(c.get('notes', []))}"
        )

    md += ["", "## Document Hygiene"]
    if not hygiene:
        md.append("- No hygiene issues detected in MVP checks.")
    for h in hygiene:
        md.append(f"- {h['item']}: **{h['status']}**. {h.get('fix','').strip()}")

    return {
        "package": package_name,
        "path": path,
        "summary": {
            "rating": rating,
            "label": label,
            "eligibility_status": eligibility_status,
        },
        # keep scorecard minimal for now; we just pass through subs
        "scorecard": {
            "subscores": subs,
        },
        "eligibility_table": eligibility,
        "criteria_findings": criteria,
        "hygiene_findings": hygiene,
        "actions": {
            "must_fix_before_submit": [
                h["item"] for h in hygiene if h.get("status") == "MISSING"
            ],
            "high_impact_improvements": [
                # still placeholder suggestions; you can refine later
                "Refine data migration plan",
                "Map resumes to key personnel requirements",
            ],
        },
        "artifacts": {
            "markdown_report": "\n".join(md),
            "cache_key": cache_key,
        },
    }

def run_review_package(path: str, options: Dict[str, Any], progress_cb: ProgressCB, session_id: str) -> Dict[str, Any]:
    start = time.time()
    docs, meta = ingest_package(path, options, progress_cb)
    requirements = extract_requirements(docs, progress_cb)
    eligibility, criteria = kb_evidence_check(requirements, progress_cb, session_id)
    hygiene = hygiene_checks(docs, progress_cb)

    # NEW: ask the model for the win verdict
    win_summary = model_win_verdict(
        path=path,
        docs=docs,
        eligibility=eligibility,
        hygiene=hygiene,
        session_id=session_id,
        progress_cb=progress_cb,
    )

    # Derive "rating" and "label" from the model verdict so the frontend doesn't break
    conf = win_summary["confidence"]
    rating = int(round(conf * 100))

    verdict = win_summary["verdict"]
    if verdict == "LIKELY_WIN":
        label = "Likely Win"
    elif verdict == "COMPETITIVE":
        label = "Competitive"
    elif verdict == "UNLIKELY":
        label = "Unlikely"
    elif verdict == "NO_GO":
        label = "No-Go"
    else:
        label = "Competitive"

    # Eligibility status: keep simple
    if any(e.get("evidence", {}).get("status") == "FAIL" for e in eligibility):
        eligibility_status = "Has FAIL items (mandatory requirements at risk)"
    else:
        eligibility_status = "All mandatory requirements satisfied or unclear"

    result = format_report(
        package_name=path.split("/")[-1],
        path=path,
        eligibility_status=eligibility_status,
        rating=rating,
        label=label,
        eligibility=eligibility,
        criteria=criteria,
        hygiene=hygiene,
        subs={"model_verdict": win_summary},
        cache_key=f"revpkg:{meta['hash']}",
    )


    _emit(progress_cb, "Complete", 1.0, {"elapsed_sec": round(time.time() - start, 2)})
    return result
