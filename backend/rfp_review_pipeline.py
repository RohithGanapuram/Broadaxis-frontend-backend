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

IMPORTANT ELIGIBILITY INTERPRETATION RULES:
- PASS = strong positive signal.
- FAIL = strong negative signal. A FAIL on a true mandatory item (insurance, signatures, required certifications) 
  should strongly reduce win probability.
- UNCLEAR = NEUTRAL.  
  It simply means "no evidence found in the BroadAxis KB", NOT that the vendor is non-compliant.  
  UNCLEAR items should:
    * NOT be treated as failures,
    * NOT significantly reduce win probability,
    * Only introduce mild uncertainty.
- UNCLEAR must never by itself result in an UNLIKELY or NO_GO verdict unless combined with a clear FAIL.
- Missing signatures or mandatory forms are considered critical hygiene issues and should meaningfully lower the score.

SCORING PRINCIPLES:
- If there are NO FAIL items and only UNCLEAR items → treat the opportunity as still viable (COMPETITIVE or POSSIBLE).
- UNCLEAR items should reduce confidence slightly (not dramatically).
- The largest penalty should come from FAIL items or missing mandatory signature forms.
- The verdict and confidence must reflect the severity of FAILs and hygiene issues, not the number of UNCLEAR items.

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
REQUIREMENT_EXTRACTOR_SYSTEM_PROMPT = """
You are an expert RFP analyst.

Goal:
Given excerpts from RFP / solicitation / scope documents, identify
VENDOR-FACING MANDATORY ELIGIBILITY REQUIREMENTS.

Eligibility requirements are things the OFFEROR / RESPONDENT / BIDDER /
VENDOR / CONTRACTOR MUST do or MUST have for the proposal to be considered:
- required forms / certifications / licenses / insurance
- minimum years of experience / project references
- required signatures / notarization
- mandatory technical or security conditions

Ignore:
- pure boilerplate legal clauses (governing law, assignment, payment terms)
- things that apply only to the agency, not the vendor.

Your output MUST be **ONLY** valid JSON with this structure:

{
  "eligibility": [
    {
      "requirement": "<verbatim or lightly cleaned requirement sentence>",
      "source_doc": "<file name or short label>",
      "notes": "<optional short note>",
      "is_hard_gate": true
    }
  ]
}

Rules:
- Do NOT include explanation text outside the JSON.
- Only include VENDOR-FACING requirements (things the proposer must do/have).
- Prefer one requirement per item (no huge multi-paragraph blobs).
- If you are unsure whether something is a hard gate, err on including it with is_hard_gate=true.
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

def is_rfp_doc(d: "Doc") -> bool:
    """
    Heuristic: treat only core RFP / scope / requirements docs as requirement sources.
    """
    name = (d.name or "").lower()
    role = (d.detected_role or "").lower()

    if any(k in name for k in [
        "rfp",
        "request for proposal",
        "solicitation",
        "scope of work",
        "scope of services",
        "statement of work",
        "sow",
        "requirements",
    ]):
        return True

    if role in {"rfp_core", "narrative"}:
        return True

    return False



import json

def extract_requirements(
    docs: List["Doc"],
    progress_cb: ProgressCB,
    session_id: str,
) -> Dict[str, Any]:
    """
    Use the LLM (via MCP) to extract eligibility requirements from the most
    RFP-like docs in this package. Falls back to regex if parsing fails.
    """
    _emit(progress_cb, "Extracting requirements…", 0.20)

    # 1) Choose candidate RFP docs
    rfp_docs = [d for d in docs if is_rfp_doc(d)]
    if not rfp_docs:
        rfp_docs = docs  # worst-case fallback

    # 2) Build a compact bundle of text to send to the model
    chunks: List[str] = []
    for d in rfp_docs[:6]:  # cap number of docs to keep tokens sane
        text = (d.text or "").strip()
        if not text:
            continue
        snippet = text[:8000]  # per-doc cap
        chunks.append(f"### DOCUMENT: {d.name}\n\n{snippet}\n")

    if not chunks:
        # No text at all – keep old structure but empty
        return {"eligibility": [], "criteria": []}

    bundle = "\n\n".join(chunks)

    user_query = (
        "Read the following RFP / scope documents and extract vendor-facing "
        "mandatory eligibility requirements as JSON in the required schema.\n\n"
        "RFP DOCUMENT EXCERPTS:\n\n"
        f"{bundle}\n"
    )

    # 3) Call MCP / Anthropic via run_mcp_query
    try:
        result = asyncio.run(run_mcp_query(
            query=user_query,
            enabled_tools=[],  # reasoning only – no tools
            model=None,        # let MCP default decide
            session_id=session_id,
            system_prompt=REQUIREMENT_EXTRACTOR_SYSTEM_PROMPT,
        ))
        raw = result.get("response", "") if isinstance(result, dict) else str(result)
    except Exception:
        raw = ""

    # 4) Try to parse JSON from the response
    data = {"eligibility": []}
    if isinstance(raw, str) and raw.strip():
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw[start:end + 1]
                data = json.loads(json_text)
        except Exception:
            data = {"eligibility": []}

    elig: List[Dict[str, Any]] = []

    # 5) If we got structured eligibility, normalize into our format
    if data.get("eligibility"):
        for idx, e in enumerate(data.get("eligibility", []), start=1):
            req_text = (e.get("requirement") or "").strip()
            if not req_text:
                continue
            source_doc = e.get("source_doc") or "RFP"
            elig.append({
                "id": f"REQ-{idx:03d}",
                "requirement": req_text[:500],
                "source_doc": source_doc,
                "required_evidence_type": None,
            })

    # 6) Fallback: if LLM / JSON failed, use the old regex approach
    if not elig:
        all_text = "\n\n".join(
            d.text for d in docs
            if d.detected_role in ("rfp_core", "narrative", "other") and d.text
        )
        for m in re.finditer(r"(?im)\b(shall|must|required|mandatory)\b.*?\.", all_text):
            t = m.group(0)
            elig.append({
                "id": f"REQ-{len(elig)+1:03d}",
                "requirement": t[:500],
                "source_doc": "RFP_Core (regex fallback)",
                "required_evidence_type": None,
            })

    # 7) Criteria – keep simple for now (same as before)
    criteria = [
        {
            "name": "Technical Approach",
            "weight": None,
            "signals": ["approach", "methodology", "architecture"],
        },
        {
            "name": "Staffing & Key Personnel",
            "weight": None,
            "signals": ["staffing", "key personnel", "org chart"],
        },
        {
            "name": "Past Performance",
            "weight": None,
            "signals": ["past performance", "experience", "references"],
        },
    ]

    return {
        "eligibility": elig,
        "criteria": criteria,
    }


def _is_vendor_mandatory(req_text: str) -> bool:
    """
    Heuristic: is this a vendor-side mandatory requirement (eligibility gate),
    rather than generic contract/legal boilerplate?
    """
    t = (req_text or "").lower()

    vendor_words = [
        "respondent", "offeror", "vendor", "contractor",
        "consultant", "proposer", "bidder", "firm", "supplier",
    ]
    trigger_phrases = [
        "must submit", "shall submit",
        "must provide", "shall provide",
        "must include", "shall include",
        "required to submit", "required to provide",
        "must have", "shall have",
        "must be licensed", "shall be licensed",
        "must demonstrate", "shall demonstrate",
        "no later than", "on or before",
    ]

    if not any(w in t for w in vendor_words):
        return False
    if not any(p in t for p in trigger_phrases):
        return False
    return True


def kb_evidence_check(
    requirements: Dict[str, Any],
    progress_cb: ProgressCB,
    session_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _emit(progress_cb, "Checking BroadAxis evidence…", 0.45)
    eligibility_out: List[Dict[str, Any]] = []

    for i, r in enumerate(requirements["eligibility"]):
        req_text = r["requirement"]

        # Skip general contract/legal boilerplate in eligibility scoring
        if not _is_vendor_mandatory(req_text):
            eligibility_out.append({
                **r,
                "evidence": {
                    "status": "N/A",
                    "kb_doc_title": None,
                    "snippet": "Treated as general contract term, not a vendor eligibility gate in this MVP.",
                    "confidence": 1.0,
                },
            })
            continue

        # Only for real vendor mandatory requirements, hit the KB
        result = asyncio.run(run_mcp_query(
            query=f"Find evidence for: {req_text}",
            enabled_tools=["Broadaxis_knowledge_search"],
            model=None,
            session_id=session_id,
            system_prompt=REVIEW_PACKAGE_SYSTEM_PROMPT,
        ))

        hits = result.get("tool_results", []) if isinstance(result, dict) else []

        if not hits:
            # ❗ No evidence found in KB → UNCLEAR, not FAIL
            score = 0.50
            status = "UNCLEAR"
            kb_doc_title = None
            snippet = "No matching evidence found in BroadAxis KB — treat as neutral/needs review. Human review required."
        else:
            best = hits[0]
            score = float(best.get("score", 0) or 0.0)
            kb_doc_title = best.get("title") or best.get("source")
            snippet = (best.get("text") or "")[:400]

            # For MVP: only PASS vs UNCLEAR. We do NOT automatically FAIL based on low score.
            if score >= 0.82:
                status = "PASS"
            else:
                status = "UNCLEAR"

        eligibility_out.append({
            **r,
            "evidence": {
                "status": status,
                "kb_doc_title": kb_doc_title,
                "snippet": snippet,
                "confidence": score,
            },
        })


        if i % 5 == 0:
            _emit(progress_cb, "Checking BroadAxis evidence…", 0.45 + 0.02 * i)

    # MVP placeholders for criteria (unchanged)
    criteria_out: List[Dict[str, Any]] = []
    for c in requirements["criteria"]:
        criteria_out.append({
            "criterion": c["name"],
            "coverage": 0.7,
            "strength": 0.6
            
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

    # 1) Attachment B check (generic, still okay)
    if not any("attachment b" in n for n in names):
        findings.append({
            "item": "Attachment B – Signature",
            "status": "MISSING",
            "fix": "Collect signature"
        })

    # 2) Only treat Form 1295 as required for TEXAS-style RFPs
    #    (we detect this from the RFP text, not just a hard-coded rule)
    combined_text = " ".join((d.text or "").lower() for d in docs[:20])
    is_texas_rfp = (
        "state of texas" in combined_text
        or "texas government code" in combined_text
        or " county, texas" in combined_text
        or "form 1295" in combined_text  # RFP itself mentions it
    )

    if is_texas_rfp and not any("form 1295" in n for n in names):
        findings.append({
            "item": "Form 1295",
            "status": "MISSING",
            "fix": "Generate and include Form 1295 (Texas Certificate of Interested Parties)"
        })

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

    # 🔹 NEW: Eligibility Gate table
        # 🔹 Eligibility Gate table (only show meaningful rows)
    md += [
        "",
        "## Eligibility Gate",
    ]

    # 1) Pick which rows to show in the markdown table
    display_eligibility: List[Dict[str, Any]] = []
    for e in eligibility or []:
        ev = e.get("evidence", {})
        status = (ev.get("status") or "").upper()

        # Only show “real” gate findings
        # - FAIL or UNCLEAR in the KB check
        # - You can add more rules here if needed
        if status in {"FAIL", "UNCLEAR"}:
            display_eligibility.append(e)

    # Optional: hard cap so UI never explodes
    MAX_ROWS = 40
    if len(display_eligibility) > MAX_ROWS:
        display_eligibility = display_eligibility[:MAX_ROWS]

    # 2) Render the markdown
    if not display_eligibility:
        if not eligibility:
            md.append("- No eligibility requirements extracted in this MVP.")
        else:
            md.append(
                "- No hard eligibility failures detected; all extracted items were "
                "treated as general contract terms in this MVP."
            )
    else:
        md.append("")
        md.append("| ID | Requirement | Status | Confidence | Evidence |")
        md.append("| --- | --- | --- | --- | --- |")
        for e in display_eligibility:
            ev = e.get("evidence", {}) or {}
            status = ev.get("status", "UNKNOWN")
            conf_ev = float(ev.get("confidence", 0.0) or 0.0)

            snippet = (ev.get("snippet") or "").replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."

            req_text = (e.get("requirement") or "").replace("|", "/")


            md.append(
                f"| {e.get('id', '-')} | {req_text} | {status} | {conf_ev:.2f} | {snippet} |"
            )


    # Existing Criteria findings section
    md += [
        "",
        "## Criteria Findings",
    ]

    for c in criteria:
        md.append(
            f"- **{c['criterion']}** — coverage {c.get('coverage', 0.0):.2f}, "
            f"strength {c.get('strength', 0.0):.2f}"
           
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
    requirements = extract_requirements(docs, progress_cb, session_id)
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

   # Eligibility status: 3-level summary (FAIL → UNCLEAR → SATISFIED)
    fail_exists = any(
        (e.get("evidence", {}).get("status") or "").upper() == "FAIL"
        for e in eligibility
    )

    unclear_exists = any(
        (e.get("evidence", {}).get("status") or "").upper() == "UNCLEAR"
        for e in eligibility
    )

    if fail_exists:
        eligibility_status = "Has FAIL items (mandatory requirements at risk)"
    elif unclear_exists:
        eligibility_status = "Has UNCLEAR items (needs human review)"
    else:
        eligibility_status = "All mandatory requirements satisfied"


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
