# backend/backend/rfp_review_pipeline.py
from typing import Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import hashlib
import time
import re

from sharepoint_api import download_package
from mcp_interface import run_mcp_query, REVIEW_PACKAGE_SYSTEM_PROMPT


ProgressCB = Callable[[str, float, Dict[str, Any]], None]

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
        tool_call = {
            "tool": "Broadaxis_knowledge_search",
            "args": {"query": query, "top_k": 5, "min_score": 0.78, "include_scores": True}
        }
        # Leverage MCP path; `tool_override` is a pattern seen in your stack to force a tool call.
        result = run_mcp_query(
            query=f"Find evidence for: {query}",
            enabled_tools=["Broadaxis_knowledge_search"],
            model=None,
            session_id=session_id,
            progress_cb=None,
            tool_override=[tool_call],
            system_prompt=REVIEW_PACKAGE_SYSTEM_PROMPT,   # <— here

        )
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

def format_report(package_name: str, path: str, eligibility_status: str, rating: int, label: str,
                  eligibility: List[Dict[str, Any]], criteria: List[Dict[str, Any]], hygiene: List[Dict[str, Any]],
                  subs: Dict[str, float], cache_key: str) -> Dict[str, Any]:
    md = [
        f"# Review Package — {package_name}",
        "",
        f"**Overall Rating:** **{rating}/100** · **{label}**",
        f"**Eligibility:** **{eligibility_status}**",
        "",
        "## Eligibility Gate",
        "| ID | Requirement | Status | Evidence | Confidence |",
        "|---|---|---|---|---|",
    ]
    for e in eligibility:
        ev = e.get("evidence", {})
        md.append(f"| {e['id']} | {e['requirement'][:80]} | {ev.get('status','-')} | {ev.get('kb_doc_title','-')} | {ev.get('confidence',0):.2f} |")

    md += [
        "",
        "## Scorecard",
        f"- Criteria coverage: **{subs['criteria_coverage']:.2f}**",
        f"- Capability fit: **{subs['capability_fit']:.2f}**",
        f"- Hygiene: **{subs['hygiene']:.2f}**",
        "",
        "## Criteria Findings",
    ]
    for c in criteria:
        md.append(f"- **{c['criterion']}** — coverage {c['coverage']:.2f}, strength {c['strength']:.2f}. Notes: {', '.join(c['notes'])}")

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
        "scorecard": {
            "weights": RUBRIC_WEIGHTS,
            "subscores": subs,
        },
        "eligibility_table": eligibility,
        "criteria_findings": criteria,
        "hygiene_findings": hygiene,
        "actions": {
            "must_fix_before_submit": [h["item"] for h in hygiene if h.get("status") == "MISSING"],
            "high_impact_improvements": ["Refine data migration plan", "Map resumes to key personnel requirements"],
        },
        "artifacts": {
            "markdown_report": "\n".join(md),
            "cache_key": cache_key,
        },
    }

def run_review_package(path: str, options: Dict[str, Any], progress_cb: ProgressCB, session_id: str) -> Dict[str, Any]:
    start = time.time()
    docs, meta = ingest_package(path, options, progress_cb)
    reqs = extract_requirements(docs, progress_cb)
    eligibility, criteria = kb_evidence_check(reqs, progress_cb, session_id)

    eligibility_status = (
        "FAIL" if any(e.get("evidence", {}).get("status") == "FAIL" for e in eligibility)
        else ("UNCLEAR" if any(e.get("evidence", {}).get("status") == "UNCLEAR" for e in eligibility) else "PASS")
    )

    hygiene = hygiene_checks(docs, progress_cb)
    rating, label, subs = score_and_label(eligibility, criteria, hygiene)

    _emit(progress_cb, "Scoring and generating report…", 0.90)
    result = format_report(
        package_name=path.split("/")[-1],
        path=path,
        eligibility_status=eligibility_status,
        rating=rating,
        label=label,
        eligibility=eligibility,
        criteria=criteria,
        hygiene=hygiene,
        subs=subs,
        cache_key=f"revpkg:{meta['hash']}",
    )

    _emit(progress_cb, "Complete", 1.0, {"elapsed_sec": round(time.time() - start, 2)})
    return result
