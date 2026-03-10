#!/usr/bin/env python3
"""generate_detailed_reports.py

Generates detailed HTML reports for candidates scoring 80+ by RE-SCORING them
with AI to get granular item-by-item breakdown.

This script:
1. Reads scored candidates from output/upload/*.csv
2. For candidates with ai_score >= 80:
   - RE-SCORES them using OpenAI with detailed prompt
   - Gets REAL scores for each compliance/must-have/nice-to-have item
   - Generates detailed scoring JSON (NO placeholders!)
   - Creates beautiful HTML report
   - Uploads HTML to Supabase (ai_report_html field)
   - Generates text embeddings and saves to Supabase (candidate_chunks)

Usage:
  python3 generate_detailed_reports.py <JOB_ID>
  python3 generate_detailed_reports.py 3419430

Output:
  - output/reports/candidate_{ID}_report.json
  - output/reports/candidate_{ID}_report.html
  - Uploads HTML to Supabase (ai_report_html)
  - Upserts vector chunks to Supabase (candidate_chunks)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Import from consolidated modules
from config import Config
from airtable_client import AirtableClient
from utils import extract_resume_text, clip


def _download_resume(url: str, candidate_id: str) -> str:
    """Download a resume from a URL, extract its text, and return it.

    Uses the Manatal auth header for authenticated URLs. Returns empty
    string on any error.
    """
    try:
        headers = {"Authorization": f"Token {Config.MANATAL_API_TOKEN}"}
        ext = Path(url.split("?")[0]).suffix.lower() or ".pdf"
        out_path = Config.OUTPUT_DIR / "resumes" / f"{candidate_id}_tier2{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        resp = requests.get(url, headers=headers, timeout=120, stream=True)
        if not resp.ok:
            print(f"  [WARN] Resume download failed ({resp.status_code}): {url[:80]}")
            return ""

        with out_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)

        text = extract_resume_text(out_path) or ""
        if text:
            print(f"  Downloaded & extracted resume from URL ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"  [WARN] Resume download error: {e}")
        return ""


# =========================
# Airtable update function
# =========================
def update_airtable_report(
    match_id: str,
    candidate_id: int,
    job_id: int,
    detailed_json: dict,
    html_path: Path,
) -> bool:
    """Upload the HTML report to Airtable and update candidate record fields.

    Finds the candidate record by match_id (formula field = job_id-candidate_id),
    uploads the HTML as an attachment to ai_report_html, and stores the
    detailed JSON string + tier2_score.
    """
    try:
        at = AirtableClient()
        html_text = html_path.read_text(encoding="utf-8")
        file_name = f"{candidate_id}_report_{job_id}.html"

        # Resolve the Airtable record ID for this candidate
        formula = f"AND({{job_id}}={job_id}, {{candidate_id}}={candidate_id})"
        records = at.get_records_by_formula(formula)
        if not records:
            print(f"[WARN] No Airtable record found for job_id={job_id} candidate_id={candidate_id}")
            return False

        record_id = records[0]["id"]
        report_score = int(detailed_json.get("overall_score") or detailed_json.get("ai_score") or 0)

        # Clear any existing report attachment before uploading the new one
        # so only one copy is ever stored in Airtable.
        at.update_record(record_id, {"ai_report_html": []})

        # Upload HTML as attachment to ai_report_html field
        at.upload_text_as_attachment(
            record_id=record_id,
            field_name="ai_report_html",
            text_content=html_text,
            filename=file_name,
        )

        # Use the rubric-driven recommendation if available; fall back to threshold
        recommendation = (detailed_json.get("recommendation") or "").strip().upper()
        if recommendation not in ("PASS", "FAIL"):
            recommendation = "PASS" if report_score >= Config.PASS_THRESHOLD else "FAIL"

        at.update_record(record_id, {
            "ai_detailed_json": json.dumps(detailed_json, ensure_ascii=False),
            "tier2_score": report_score,
            "tier2_status": recommendation,
            "ai_summary": detailed_json.get("ai_summary") or "",
            "ai_strengths": detailed_json.get("ai_strengths") or "",
            "ai_gaps": detailed_json.get("ai_gaps") or "",
        })

        return True

    except Exception as e:
        print(f"[WARN] Airtable report update failed: {e}")
        return False


# =========================
# Rubric loading / parsing
# =========================
def load_rubric_json(job_id: str) -> dict:
    """Load rubric JSON for a job.

    Tries Airtable first (primary), then falls back to local rubric file.
    """
    at = AirtableClient()
    rubric = at.get_rubric(job_id)
    if rubric:
        return rubric

    # Fallback: local rubric file
    rubric_path = Config.get_rubric_path(job_id)
    if rubric_path.exists():
        print(f"  [fallback] Loaded rubric from local file: {rubric_path}")
        return json.loads(rubric_path.read_text(encoding="utf-8"))

    raise LookupError(f"No rubric found for job_id={job_id!r} (checked Airtable and local file)")


def parse_rubric_structure(rubric: dict) -> Dict[str, Any]:
    """Extract structured rubric data.

    Supports both schemas:
    - New JSON schema: compliance_requirements (list of strings),
      requirements.must_have (list with id/evidence_signals/negative_signals),
      semantic_ontology.normalized_terms (list of strings)
    - Legacy schema: compliance (list of dicts), must_have/nice_to_have (top-level),
      normalized_terms (dict of objects)

    Pass threshold is always sourced from Config.PASS_THRESHOLD (user-configured),
    never from the rubric.
    """
    compliance: List[dict] = []
    must_have: List[dict] = []
    nice_to_have: List[dict] = []
    semantic_terms: List[str] = []

    # ===== COMPLIANCE PARSING =====
    # New schema: compliance_requirements is a flat list of strings
    for comp in rubric.get("compliance_requirements", []):
        if isinstance(comp, str) and comp:
            compliance.append({"item": comp, "details": ""})
        elif isinstance(comp, dict):
            compliance.append({"item": comp.get("item", comp.get("requirement", "")), "details": comp.get("details", "")})

    # Legacy schema: top-level compliance list of dicts
    if not compliance:
        for comp in rubric.get("compliance", []):
            if isinstance(comp, dict):
                compliance.append({"item": comp.get("item", comp.get("requirement", "")), "details": comp.get("details", "")})
            elif isinstance(comp, str) and comp:
                compliance.append({"item": comp, "details": ""})

    # ===== MUST-HAVE PARSING =====
    # New schema: requirements.must_have with id, evidence_signals, negative_signals
    requirements = rubric.get("requirements", {})
    mh_source = requirements.get("must_have", []) if isinstance(requirements, dict) else []
    # Legacy schema: top-level must_have
    if not mh_source:
        mh_source = rubric.get("must_have", rubric.get("must_haves", []))

    for i, req in enumerate(mh_source, 1):
        if isinstance(req, dict):
            must_have.append({
                "id": req.get("id", f"MH{i}"),
                "requirement": req.get("requirement", ""),
                "weight": float(req.get("weight", 0)),
                "evidence_signals": req.get("evidence_signals", []),
                "negative_signals": req.get("negative_signals", []),
            })

    # ===== NICE-TO-HAVE PARSING =====
    # New schema: requirements.nice_to_have
    nth_source = requirements.get("nice_to_have", []) if isinstance(requirements, dict) else []
    # Legacy schema: top-level nice_to_have
    if not nth_source:
        nth_source = rubric.get("nice_to_have", rubric.get("nice_to_haves", []))

    for i, skill in enumerate(nth_source, 1):
        if isinstance(skill, dict):
            nice_to_have.append({
                "id": skill.get("id", f"NH{i}"),
                "skill": skill.get("skill", ""),
                "weight": float(skill.get("weight", 0)),
            })

    # ===== SEMANTIC TERMS =====
    # New schema: semantic_ontology.normalized_terms (list of strings)
    sem = rubric.get("semantic_ontology", {})
    if isinstance(sem, dict):
        for t in sem.get("normalized_terms", []):
            if isinstance(t, str) and t:
                semantic_terms.append(t)
    # Legacy schema: top-level normalized_terms (dict of objects)
    if not semantic_terms:
        for term, details in rubric.get("normalized_terms", {}).items():
            semantic_terms.append(term)
            if isinstance(details, dict):
                semantic_terms.extend(a for a in details.get("aliases", [])[:3] if a)

    # Pass threshold is user-input derived (Config.PASS_THRESHOLD); never read from rubric.
    return {
        "compliance": compliance,
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "semantic_terms": semantic_terms,
        "pass_threshold": Config.PASS_THRESHOLD,
    }


# =========================
# AI DETAILED SCORING PROMPT (merged: python10c primary + python8 additions)
# =========================
def build_detailed_scoring_prompt(rubric: dict, rubric_structure: Dict[str, Any], resume_text: str, rubric_version: str) -> str:
    """Build Tier 2 detailed scoring prompt.

    Merges python10c (primary: item IDs, evidence/negative signals, 0-5 scale,
    contribution formula) with python8 additions (ROLE/JD header, semantic
    guidance, CRITICAL rubric-only instructions, exact item counts, bias guardrails).
    """
    role = rubric.get("role", rubric.get("role_applied", ""))
    jd_obj = rubric.get("jd", {})
    jd_summary = jd_obj.get("jd_summary", rubric.get("jd_summary", "")) if isinstance(jd_obj, dict) else rubric.get("jd_summary", "")
    bias_guardrails = rubric.get("bias_guardrails", [])

    must_have_reqs = rubric_structure.get("must_have", [])
    nice_to_have_skills = rubric_structure.get("nice_to_have", [])
    compliance_items = rubric_structure.get("compliance", [])
    semantic_terms = rubric_structure.get("semantic_terms", [])
    pass_threshold = rubric_structure.get("pass_threshold", Config.PASS_THRESHOLD)

    mh_ids = [r.get("id", f"MH{i+1}") for i, r in enumerate(must_have_reqs)]
    nh_ids = [s.get("id", f"NH{i+1}") for i, s in enumerate(nice_to_have_skills)]

    prompt = (
        "You are an expert technical recruiter. Evaluate this candidate STRICTLY against the rubric.\n\n"
        "CRITICAL: The rubric is the SINGLE SOURCE OF TRUTH. Do not add criteria beyond what is stated. "
        "Do not invent requirements not listed.\n\n"
        f"RUBRIC_VERSION: {rubric_version}\n"
        f"ROLE: {role}\n"
        f"JOB SUMMARY: {jd_summary}\n\n"
    )

    # Semantic guidance (from python8) — helps recognise synonyms
    if semantic_terms:
        prompt += "SEMANTIC GUIDANCE (accept equivalent terms for these):\n"
        prompt += ", ".join(semantic_terms[:30]) + "\n\n"

    # Compliance requirements (as defined in the rubric — PASS/FAIL only)
    if compliance_items:
        prompt += f"COMPLIANCE REQUIREMENTS (PASS/FAIL — {len(compliance_items)} item(s) as defined in rubric):\n"
        for i, item in enumerate(compliance_items, 1):
            prompt += f"{i}. {item.get('item', '')}\n"
        prompt += "\n"

    # Must-have requirements with IDs and signals (python10c style)
    if must_have_reqs:
        prompt += f"MUST-HAVE REQUIREMENTS — use EXACTLY these IDs in order: {mh_ids}\n"
        for req in must_have_reqs:
            rid = req.get("id", "")
            requirement = req.get("requirement", "")
            weight = req.get("weight", 0)
            signals = req.get("evidence_signals", [])
            neg_signals = req.get("negative_signals", [])
            prompt += f"\n{rid}: {requirement} (weight: {weight}%)\n"
            if signals:
                prompt += f"  Evidence signals: {'; '.join(signals)}\n"
            if neg_signals:
                prompt += f"  Red flags: {'; '.join(neg_signals)}\n"
        prompt += "\n"

    # Nice-to-have skills
    if nice_to_have_skills:
        prompt += f"NICE-TO-HAVE SKILLS — use EXACTLY these IDs in order: {nh_ids}\n"
        for skill in nice_to_have_skills:
            sid = skill.get("id", "")
            skill_text = skill.get("skill", "")
            weight = skill.get("weight", 0)
            prompt += f"{sid}: {skill_text} (weight: {weight}%)\n"
        prompt += "\n"

    prompt += f"CANDIDATE RESUME:\n{clip(resume_text, Config.MAX_RESUME_CHARS)}\n\n"

    # Scoring scale: prefer rubric-defined labels when available, otherwise fall back to linear defaults
    scoring = rubric.get("scoring", {}) if isinstance(rubric, dict) else {}
    scale = scoring.get("scale") if isinstance(scoring, dict) else None
    if isinstance(scale, dict) and scale:
        try:
            scale_pairs = sorted(((int(k), str(v)) for k, v in scale.items()), key=lambda x: x[0])
        except (TypeError, ValueError):
            scale_pairs = []
    else:
        scale_pairs = []

    if scale_pairs:
        max_score = max(k for k, _ in scale_pairs)
        prompt += f"SCORING SCALE (0-{max_score}, from rubric):\n"
        for k, label in scale_pairs:
            prompt += f"{k} = {label}\n"
        prompt += "\n"
    else:
        # Fall back only if rubric does not define a scale; default to 0-5
        max_score = 5
        prompt += (
            "SCORING SCALE (0-5, linear — 5 = fully meets requirement):\n"
            "0 = No evidence\n"
            "1 = Minimal; little or no relevant evidence\n"
            "2 = Partial; some evidence but significant gaps\n"
            "3 = Adequate; meets requirement with sufficient evidence\n"
            "4 = Strong; clearly meets requirement with good evidence\n"
            "5 = Full; fully meets requirement with clear, specific evidence\n\n"
        )

    prompt += (
        f"SCORING FORMULA: contribution = (score / {max_score}) × weight\n"
        "overall_score = sum of all contributions (normalised to 0-100)\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        f"1. The must_have array MUST contain EXACTLY {len(must_have_reqs)} items using IDs {mh_ids}.\n"
        f"2. The nice_to_have array MUST contain EXACTLY {len(nice_to_have_skills)} items using IDs {nh_ids}.\n"
        f"3. The compliance array MUST contain EXACTLY {len(compliance_items)} item(s) matching the rubric above.\n"
        "4. Use the EXACT item ID, requirement text, and weight from the rubric. Do NOT rephrase or add criteria.\n"
        "5. Provide specific evidence quotes from the resume for each score.\n"
        f"6. Pass threshold: overall_score >= {pass_threshold}. Set recommendation to PASS or FAIL accordingly.\n"
        f"7. overall_score MUST equal sum of (score/{max_score} × weight) for all items.\n"
        "8. Evaluate on demonstrated outcomes only. Ignore protected attributes (age, gender, race, etc.).\n"
        "9. Respond with ONLY valid JSON. No markdown, no code blocks, no preamble.\n"
    )

    if bias_guardrails:
        guardrail_texts = [str(g) for g in (bias_guardrails if isinstance(bias_guardrails, list) else [])[:3]]
        if guardrail_texts:
            prompt += f"\nBIAS GUARDRAILS: {' '.join(guardrail_texts)}\n"

    prompt += """
REQUIRED JSON OUTPUT:
{
  "compliance": [{"requirement": "EXACT text", "status": "PASS|FAIL", "evidence": "brief reason"}],
  "must_have": [{"id": "MHx", "requirement": "EXACT text", "score": 0-5, "weight": N, "contribution": N, "evidence": "specific quote"}],
  "nice_to_have": [{"id": "NHx", "skill": "EXACT text", "score": 0-5, "weight": N, "contribution": N, "evidence": "specific quote"}],
  "overall_score": N,
  "ai_score": N,
  "ai_summary": "2-3 sentence assessment (max 60 words)",
  "ai_strengths": "comma-separated strengths",
  "ai_gaps": "comma-separated gaps",
  "recommendation": "PASS|FAIL",
  "floor_triggered": false
}"""

    return prompt


def llm_score_detailed(
    oa: OpenAI,
    rubric: dict,
    rubric_structure: Dict[str, Any],
    rubric_version: str,
    resume_text: str
) -> Dict[str, Any]:
    """Score candidate with detailed item-by-item breakdown using AI.
    
    CRITICAL: Uses parsed rubric_structure (normalized format) as source of truth.
    
    Returns dict with complete detailed scoring (NO placeholders).
    """
    
    prompt = build_detailed_scoring_prompt(rubric, rubric_structure, resume_text, rubric_version)
    
    try:
        r = oa.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a technical recruiter. Respond only with valid JSON. No markdown, no code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        
        text = r.choices[0].message.content or ""
        
        # Clean markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            if text.startswith("json"):
                text = text[4:].strip()
        
        # Try to parse JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from text
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                data = json.loads(m.group(0))
            else:
                raise ValueError("Could not extract valid JSON from response")
        
        # Validate that we have the required fields
        if "ai_score" not in data:
            data["ai_score"] = data.get("overall_score", 0)
        if "ai_summary" not in data:
            data["ai_summary"] = data.get("summary", "No summary provided")
        if "ai_strengths" not in data:
            strengths = data.get("strengths", [])
            data["ai_strengths"] = ", ".join(strengths) if isinstance(strengths, list) else str(strengths)
        if "ai_gaps" not in data:
            gaps = data.get("gaps", [])
            data["ai_gaps"] = ", ".join(gaps) if isinstance(gaps, list) else str(gaps)
        
        return data
        
    except Exception as e:
        print(f"      ✗ Detailed scoring error: {e}")
        # Return error response
        return {
            "compliance": [],
            "must_have": [],
            "nice_to_have": [],
            "overall_score": 0,
            "ai_score": 0,
            "ai_summary": f"ERROR: {str(e)[:100]}",
            "ai_strengths": "",
            "ai_gaps": "Scoring failed",
            "recommendation": "FAIL",
            "floor_triggered": False
        }


# =========================
# Normalize LLM response to match rubric (count + item text)
# =========================
def normalize_detailed_response(ai_data: dict, rubric_structure: Dict[str, Any]) -> dict:
    """Ensure compliance, must_have, and nice_to_have match rubric exactly.

    - Same number of items as rubric.
    - Item descriptions (requirement/skill/item text) exactly from rubric.
    - Scores, status, evidence from LLM response matched by index or id; missing => 0 / NOT_ASSESSED / "".
    """
    out = dict(ai_data)

    # ── Compliance ─────────────────────────────────────────────────────────
    rubric_compliance = rubric_structure.get("compliance", [])
    ai_compliance = ai_data.get("compliance", []) or []
    norm_compliance = []
    for i, r_comp in enumerate(rubric_compliance):
        item_text = r_comp.get("item", "") if isinstance(r_comp, dict) else str(r_comp)
        entry = {"requirement": item_text, "item": item_text, "status": "NOT_ASSESSED", "evidence": "", "details": ""}
        if i < len(ai_compliance):
            a = ai_compliance[i]
            if isinstance(a, dict):
                entry["status"] = a.get("status", "NOT_ASSESSED")
                entry["evidence"] = a.get("evidence", a.get("details", ""))
                entry["details"] = entry["evidence"]
        norm_compliance.append(entry)
    out["compliance"] = norm_compliance

    # ── Must-have: match by index first, then by id ──────────────────────────
    rubric_mh = rubric_structure.get("must_have", [])
    ai_mh = ai_data.get("must_have", []) or []
    ai_mh_by_id = {str(x.get("id", "")): x for x in ai_mh if x.get("id")}
    norm_mh = []
    for i, r in enumerate(rubric_mh):
        if not isinstance(r, dict):
            continue
        rid = r.get("id", f"MH{i+1}")
        req = r.get("requirement", "")
        weight = float(r.get("weight", 0))
        entry = {"id": rid, "requirement": req, "weight": weight, "score": 0, "evidence": "", "contribution": 0}
        if i < len(ai_mh) and isinstance(ai_mh[i], dict):
            a = ai_mh[i]
            entry["score"] = a.get("score", 0)
            entry["evidence"] = a.get("evidence", "")
            entry["contribution"] = a.get("contribution", 0)
        elif rid in ai_mh_by_id:
            a = ai_mh_by_id[rid]
            entry["score"] = a.get("score", 0)
            entry["evidence"] = a.get("evidence", "")
            entry["contribution"] = a.get("contribution", 0)
        norm_mh.append(entry)
    out["must_have"] = norm_mh

    # ── Nice-to-have: match by index first, then by id ───────────────────────
    rubric_nth = rubric_structure.get("nice_to_have", [])
    ai_nth = ai_data.get("nice_to_have", []) or []
    ai_nth_by_id = {str(x.get("id", "")): x for x in ai_nth if x.get("id")}
    norm_nth = []
    for i, r in enumerate(rubric_nth):
        if not isinstance(r, dict):
            continue
        rid = r.get("id", f"NH{i+1}")
        skill = r.get("skill", "")
        weight = float(r.get("weight", 0))
        entry = {"id": rid, "skill": skill, "weight": weight, "score": 0, "evidence": "", "contribution": 0}
        if i < len(ai_nth) and isinstance(ai_nth[i], dict):
            a = ai_nth[i]
            entry["score"] = a.get("score", 0)
            entry["evidence"] = a.get("evidence", "")
            entry["contribution"] = a.get("contribution", 0)
        elif rid in ai_nth_by_id:
            a = ai_nth_by_id[rid]
            entry["score"] = a.get("score", 0)
            entry["evidence"] = a.get("evidence", "")
            entry["contribution"] = a.get("contribution", 0)
        norm_nth.append(entry)
    out["nice_to_have"] = norm_nth

    return out


# =========================
# Server-side score recomputation (rubric-derived scale)
# =========================
def _recompute_score(data: dict, rating_max: float) -> float:
    """Recalculate overall_score in Python to override LLM arithmetic.

    Uses the rubric-defined rating scale: contribution = (score / rating_max) × weight.
    Caps at 100.0.
    """
    total = 0.0
    for item in data.get("must_have", []):
        s = float(item.get("score", 0) or 0)
        w = float(item.get("weight", 0) or 0)
        contrib = round((s / float(rating_max or 1.0)) * w, 4)
        item["contribution"] = contrib
        total += contrib
    for item in data.get("nice_to_have", []):
        s = float(item.get("score", 0) or 0)
        w = float(item.get("weight", 0) or 0)
        contrib = round((s / float(rating_max or 1.0)) * w, 4)
        item["contribution"] = contrib
        total += contrib
    return round(min(total, 100.0), 1)


# =========================
# Enhanced JSON generation with AI scoring
# =========================
def generate_detailed_json_with_ai(
    candidate: Dict[str, Any],
    rubric: dict,
    rubric_structure: Dict[str, Any],
    resume_text: str,
    openai_client: OpenAI
) -> Dict[str, Any]:
    """Generate detailed JSON by re-scoring with AI for granular breakdown.

    After the LLM returns, applies server-side score recomputation and the
    hard-coded floor rule so that final scores and recommendations are reliable.
    """
    print(f"    Re-scoring with AI for detailed breakdown...")

    # Version is in metadata.version for new JSON schema; fall back to top-level
    rubric_version = str(
        rubric.get("metadata", {}).get("version")
        or rubric.get("version", "1.0")
    )
    ai_data = llm_score_detailed(openai_client, rubric, rubric_structure, rubric_version, resume_text)

    # ── Normalize so report items 100% match rubric (count + descriptions) ───
    ai_data = normalize_detailed_response(ai_data, rubric_structure)

    # Determine rating scale max from rubric scoring.scale (fallback 5)
    rating_max = 5
    scoring = rubric.get("scoring", {}) if isinstance(rubric, dict) else {}
    scale = scoring.get("scale") if isinstance(scoring, dict) else None
    if isinstance(scale, dict) and scale:
        try:
            keys = [int(k) for k in scale.keys()]
            if keys:
                rating_max = max(keys)
        except (TypeError, ValueError):
            pass

    # ── Server-side score recomputation ──────────────────────────────────────
    recomputed = _recompute_score(ai_data, rating_max)

    # ── Hard-coded floor rule ─────────────────────────────────────────────────
    pass_threshold = rubric_structure.get("pass_threshold", Config.PASS_THRESHOLD)
    floor_triggered = any(
        float(item.get("score", 0) or 0) < 2
        for item in ai_data.get("must_have", [])
    )
    recommendation = "FAIL" if (floor_triggered or recomputed < pass_threshold) else "PASS"

    ai_data["floor_triggered"] = floor_triggered
    ai_data["recommendation"] = recommendation
    ai_data["overall_score"] = recomputed
    ai_data["ai_score"] = int(round(recomputed))

    # Parse strengths and gaps into lists
    ai_strengths = ai_data.get("ai_strengths", "")
    ai_gaps = ai_data.get("ai_gaps", "")
    strengths_list = [s.strip() for s in ai_strengths.split(",") if s.strip()]
    gaps_list = [g.strip() for g in ai_gaps.split(",") if g.strip()]

    # Calculate actual weights from rubric (not hardcoded)
    must_have_total_weight = sum(float(item.get("weight", 0)) for item in rubric_structure["must_have"])
    nice_to_have_total_weight = sum(float(item.get("weight", 0)) for item in rubric_structure["nice_to_have"])

    company_name = rubric.get("company", "Recruitment System")

    detailed_json = {
        "compliance": ai_data.get("compliance", []),
        "must_have": ai_data.get("must_have", []),
        "nice_to_have": ai_data.get("nice_to_have", []),
        "overall_score": recomputed,
        "ai_score": int(round(recomputed)),
        "ai_summary": ai_data.get("ai_summary", ""),
        "ai_strengths": ai_strengths,
        "ai_gaps": ai_gaps,
        "recommendation": recommendation,
        "floor_triggered": floor_triggered,
        "candidate_name": candidate.get("full_name", ""),
        "candidate_id": candidate.get("candidate_id", ""),
        "position": candidate.get("job_name", ""),
        "pass_threshold": pass_threshold,
        "report_date": datetime.now().strftime("%B %d, %Y"),
        "generated_at": datetime.now().isoformat(),
        "generated_by": f"{company_name} Recruitment System",
        "key_strengths": strengths_list,
        "development_areas": gaps_list,
        "must_have_weight": int(must_have_total_weight),
        "nice_to_have_weight": int(nice_to_have_total_weight),
        "rating_max": rating_max,
    }

    return detailed_json


# =========================
# HTML report generation (matches Candidate Evaluation reference design)
# =========================
def generate_html_report(detailed_json: Dict[str, Any]) -> str:
    """Generate HTML report matching the reference Candidate Evaluation design.

    Layout and section order follow the gold-standard: Candidate Evaluation - Shailesh Waren.html
    """

    candidate_name = detailed_json.get("candidate_name", "Candidate")
    position = detailed_json.get("position", "Position")
    overall_score = detailed_json.get("overall_score", 0)
    recommendation = detailed_json.get("recommendation", "REVIEW")
    report_date = detailed_json.get("report_date", datetime.now().strftime("%B %d, %Y"))
    executive_summary = detailed_json.get("ai_summary", "")

    key_strengths = detailed_json.get("key_strengths", [])
    development_areas = detailed_json.get("development_areas", [])
    compliance = detailed_json.get("compliance", [])
    must_have = detailed_json.get("must_have", [])
    nice_to_have = detailed_json.get("nice_to_have", [])
    must_have_weight = detailed_json.get("must_have_weight", 90)
    nice_to_have_weight = detailed_json.get("nice_to_have_weight", 10)
    rating_max = detailed_json.get("rating_max", 5)

    if recommendation == "PASS":
        badge_class = "badge-pass"
    elif recommendation == "REVIEW":
        badge_class = "badge-review"
    else:
        badge_class = "badge-fail"

    def score_color(score: float, max_score: float = rating_max) -> str:
        ratio = score / max_score if max_score > 0 else 0
        if ratio >= 1.0:
            return "#15803d"   # Full (max score)
        elif ratio >= 0.8:
            return "#16a34a"   # Strong
        elif ratio >= 0.6:
            return "#ca8a04"   # Adequate
        else:
            return "#dc2626"   # Partial/None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Candidate Evaluation - {candidate_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #1f2937; background: #f9fafb; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 30px; }}
        .header {{ margin-bottom: 30px; }}
        .candidate-name {{ font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 5px; }}
        .position-title {{ font-size: 14px; color: #6b7280; margin-bottom: 20px; }}
        .info-boxes {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 30px; }}
        .info-box {{ background: #f3f4f6; padding: 15px; border-radius: 6px; }}
        .info-label {{ font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
        .info-value {{ font-size: 16px; font-weight: 600; color: #111827; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; color: white; }}
        .badge-pass {{ background: #16a34a; }}
        .badge-fail {{ background: #dc2626; }}
        .badge-review {{ background: #ea580c; }}
        .score-large {{ font-size: 32px; font-weight: 700; color: #111827; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 12px; }}
        .section-content {{ font-size: 14px; color: #4b5563; line-height: 1.7; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }}
        thead {{ background: #9ca3af; color: white; }}
        thead.must-have {{ background: #3b82f6; }}
        thead.nice-to-have {{ background: #3b82f6; }}
        th {{ padding: 10px; text-align: left; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }}
        td {{ padding: 12px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
        tr:last-child td {{ border-bottom: none; }}
        .requirement-text {{ font-weight: 500; color: #111827; margin-bottom: 3px; }}
        .score-cell {{ font-weight: 700; font-size: 14px; }}
        .weight-cell {{ color: #6b7280; }}
        .evidence-text {{ font-size: 12px; color: #6b7280; font-style: italic; }}
        .tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 15px 0; }}
        .tag {{ background: #16a34a; color: white; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
        .tag-red {{ background: #dc2626; }}
        .interview-questions {{ background: #f9fafb; padding: 20px; border-radius: 6px; border-left: 4px solid #3b82f6; }}
        .interview-questions ul {{ list-style: none; padding-left: 0; }}
        .interview-questions li {{ padding: 10px 0; padding-left: 25px; position: relative; font-size: 14px; color: #374151; }}
        .interview-questions li:before {{ content: "•"; position: absolute; left: 10px; color: #3b82f6; font-weight: bold; font-size: 18px; }}
        @media print {{ body {{ background: white; }} .container {{ box-shadow: none; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="candidate-name">{candidate_name}</h1>
            <p class="position-title">Candidate Evaluation Report</p>
        </div>

        <div class="info-boxes">
            <div class="info-box">
                <div class="info-label">Position</div>
                <div class="info-value">{position}</div>
                <div style="margin-top: 8px;">
                    <span class="badge {badge_class}">{recommendation}</span>
                </div>
            </div>
            <div class="info-box">
                <div class="info-label">Overall Score</div>
                <div class="score-large">{overall_score}</div>
            </div>
            <div class="info-box">
                <div class="info-label">Report Date</div>
                <div class="info-value">{report_date}</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Executive Summary</h2>
            <p class="section-content">{executive_summary}</p>
        </div>
"""

    # Compliance section
    if compliance:
        html += """
        <div class="section">
            <h2 class="section-title">Compliance Requirements</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50%;">Requirement</th>
                        <th style="width: 15%;">Status</th>
                        <th style="width: 35%;">Details</th>
                    </tr>
                </thead>
                <tbody>
"""
        for item in compliance:
            req_text = item.get("requirement", item.get("item", ""))
            status = item.get("status", "NOT_ASSESSED")
            details = item.get("evidence", item.get("details", ""))
            if status == "PASS":
                b_cls, b_lbl = "badge-pass", "COMPLY"
            elif status == "FAIL":
                b_cls, b_lbl = "badge-fail", "REVIEW"
            else:
                b_cls, b_lbl = "badge-review", status
            html += f"""
                    <tr>
                        <td class="requirement-text">{req_text}</td>
                        <td><span class="badge {b_cls}">{b_lbl}</span></td>
                        <td class="evidence-text">{details}</td>
                    </tr>"""
        html += """
                </tbody>
            </table>
        </div>"""

    # Must-have section
    if must_have:
        html += f"""
        <div class="section">
            <h2 class="section-title">Must-Have Requirements ({must_have_weight}% Weight)</h2>
            <table>
                <thead class="must-have">
                    <tr>
                        <th style="width: 40%;">Requirement</th>
                        <th style="width: 10%;">Score</th>
                        <th style="width: 10%;">Weight</th>
                        <th style="width: 40%;">Evidence</th>
                    </tr>
                </thead>
                <tbody>"""
        for item in must_have:
            score = float(item.get("score", 0))
            weight = item.get("weight", 0)
            evidence = item.get("evidence", "")
            color = score_color(score)
            html += f"""
                    <tr>
                        <td class="requirement-text">{item.get("requirement", "")}</td>
                        <td class="score-cell" style="color: {color};">{int(score)}/{rating_max}</td>
                        <td class="weight-cell">{weight}%</td>
                        <td class="evidence-text">{evidence}</td>
                    </tr>"""
        html += """
                </tbody>
            </table>
        </div>"""

    # Nice-to-have section
    if nice_to_have:
        html += f"""
        <div class="section">
            <h2 class="section-title">Nice-to-Have Skills ({nice_to_have_weight}% Weight)</h2>
            <table>
                <thead class="nice-to-have">
                    <tr>
                        <th style="width: 40%;">Skill</th>
                        <th style="width: 10%;">Score</th>
                        <th style="width: 10%;">Weight</th>
                        <th style="width: 40%;">Evidence</th>
                    </tr>
                </thead>
                <tbody>"""
        for item in nice_to_have:
            score = float(item.get("score", 0))
            weight = item.get("weight", 0)
            evidence = item.get("evidence", "")
            color = score_color(score)
            html += f"""
                    <tr>
                        <td class="requirement-text">{item.get("skill", "")}</td>
                        <td class="score-cell" style="color: {color};">{int(score)}/{rating_max}</td>
                        <td class="weight-cell">{weight}%</td>
                        <td class="evidence-text">{evidence}</td>
                    </tr>"""
        html += """
                </tbody>
            </table>
        </div>"""

    # Assessment Summary
    html += """
        <div class="section">
            <h2 class="section-title">Assessment Summary</h2>"""
    if key_strengths:
        html += """
            <h3 style="font-size: 14px; font-weight: 600; margin: 15px 0 10px 0;">Key Strengths</h3>
            <div class="tags">"""
        for s in key_strengths:
            html += f'\n                <span class="tag">{s}</span>'
        html += "\n            </div>"
    if development_areas:
        html += """
            <h3 style="font-size: 14px; font-weight: 600; margin: 20px 0 10px 0;">Gap</h3>
            <p class="section-content" style="margin-bottom: 10px;">Missing skills or experience in key areas</p>
            <div class="tags">"""
        for a in development_areas:
            html += f'\n                <span class="tag tag-red">{a}</span>'
        html += "\n            </div>"
    html += "\n        </div>"

    # Suggested Interview Questions
    interview_questions: List[str] = []
    for item in must_have[:5]:
        req = item.get("requirement", "").rstrip(".")
        score = float(item.get("score", 0))
        req_lower = req[0].lower() + req[1:] if req else req
        if score >= 3:
            interview_questions.append(
                f"Can you walk us through your experience with {req_lower}? Please provide specific examples."
            )
        elif score >= 1:
            interview_questions.append(
                f"We noticed some experience with {req_lower}. Can you elaborate on your hands-on work in this area?"
            )
        else:
            interview_questions.append(
                f"This role requires {req_lower}. How would you approach developing this skill if given the opportunity?"
            )
    if position:
        interview_questions.append(
            f"What interests you most about this {position} role, and how do your skills align with our requirements?"
        )

    if interview_questions:
        html += """
        <div class="section">
            <h2 class="section-title">Suggested Interview Questions</h2>
            <div class="interview-questions">
                <ul>"""
        for q in interview_questions:
            html += f"\n                    <li>{q}</li>"
        html += """
                </ul>
            </div>
        </div>"""

    html += f"""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #9ca3af; text-align: center;">
            <p>Generated on {report_date} &bull; AI-Powered Candidate Evaluation</p>
        </div>
    </div>
</body>
</html>"""

    return html


# =========================
# Helper functions
# =========================
def load_job_description(job_id: str) -> str:
    """Load job description from various sources (in priority order)."""
    
    # Priority 1: Dedicated JD file for this job
    jd_file_specific = Path(f"offline_input/jd_{job_id}.txt")
    if jd_file_specific.exists():
        return jd_file_specific.read_text(encoding="utf-8")
    
    # Priority 2: Generic JD file
    jd_file_generic = Path("offline_input/jd.txt")
    if jd_file_generic.exists():
        return jd_file_generic.read_text(encoding="utf-8")
    
    # Priority 3: JD embedded in offline job JSON
    offline_json = Path(f"offline_input/job_{job_id}.json")
    if offline_json.exists():
        try:
            import json
            with offline_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
                jd_text = data.get("jd_text", "")
                if jd_text:
                    print(f"  📄 Loaded JD from {offline_json}")
                    return jd_text
        except Exception as e:
            print(f"  ⚠ Failed to read JD from JSON: {e}")
    
    # No JD found
    print(f"  ⚠ No JD found for job {job_id}")
    return ""


def get_resume_path(candidate: Dict[str, Any]) -> Optional[Path]:
    """Get resume path from candidate data."""
    
    candidate_id = candidate.get("candidate_id", "")
    
    # Check if resume_local_path is in candidate data
    resume_local = candidate.get("resume_local_path", "")
    if resume_local:
        path = Path(resume_local)
        if path.exists():
            return path
    
    # Check output/resumes folder
    resumes_dir = Config.OUTPUT_DIR / "resumes"
    if resumes_dir.exists():
        for ext in [".pdf", ".docx", ".doc"]:
            path = resumes_dir / f"{candidate_id}{ext}"
            if path.exists():
                return path
    
    # Check offline_input/resumes folder
    offline_resumes = Path("offline_input/resumes")
    if offline_resumes.exists():
        for file in offline_resumes.glob(f"*{candidate_id}*"):
            if file.suffix.lower() in ['.pdf', '.docx', '.doc']:
                return file
        
        # Try by name
        full_name = candidate.get("full_name", "")
        if full_name:
            name_safe = re.sub(r'[^\w\s-]', '', full_name).replace(' ', '_')
            for file in offline_resumes.glob(f"*{name_safe}*"):
                if file.suffix.lower() in ['.pdf', '.docx', '.doc']:
                    return file
    
    return None


# =========================
# Main
# =========================
def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 generate_detailed_reports.py <JOB_ID>")
        return 2
    
    job_id = sys.argv[1].strip()
    
    try:
        Config.validate()
        openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    Config.ensure_dirs()
    
    scored_csv = Config.get_scored_csv_path(job_id)
    if not scored_csv.exists():
        print(f"ERROR: Scored CSV not found: {scored_csv}")
        print("Run python8.py first to generate scored data.")
        return 2
    
    try:
        rubric = load_rubric_json(job_id)
        print(f"Loaded rubric for job_id={job_id}")
        rubric_structure = parse_rubric_structure(rubric)
    except Exception as e:
        print(f"ERROR: Failed to load rubric: {e}")
        return 2
    
    # Note: JD text is NOT loaded here - rubric is the single source of truth for scoring
    # JD is only stored in job_3419430.json for recruiter reference
    
    with scored_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        candidates = list(reader)
    
    total_candidates = len(candidates)
    min_score = Config.MIN_SCORE_FOR_REPORT
    # Prefer tier1_score (set by python8.py Tier 1 pass); fall back to ai_score for legacy CSVs
    high_scorers = [
        c for c in candidates
        if float(c.get("tier1_score") or c.get("ai_score") or 0) >= min_score
    ]
    
    generated_count = 0
    uploaded_count = 0

    # Build Airtable lookup: cache_key → True if report already uploaded
    print(f"Fetching existing Airtable records for job_id={job_id}...")
    try:
        at_check = AirtableClient()
        existing_records = at_check.get_records_by_formula(f"{{job_id}}={job_id}")
        airtable_has_report: Dict[str, bool] = {
            r["fields"]["cache_key"]: bool(r["fields"].get("ai_report_html"))
            for r in existing_records
            if r["fields"].get("cache_key")
        }
        skippable = sum(1 for v in airtable_has_report.values() if v)
        print(f"Found {len(airtable_has_report)} candidate record(s), {skippable} already have a report\n")
    except Exception as e:
        print(f"[WARN] Could not fetch Airtable records for skip check: {e}\n")
        airtable_has_report = {}

    print(f"\n{'='*70}")
    print(f"Generating Detailed AI-Powered Reports for Job {job_id}")
    print(f"{'='*70}")
    print(f"Total candidates: {total_candidates}")
    print(f"Candidates scoring ≥ {min_score}: {len(high_scorers)}")
    print(f"{'='*70}\n")

    
    for candidate in high_scorers:
        ai_score = float(candidate.get("ai_score", 0))
        candidate_id = candidate.get("candidate_id", "")
        full_name = candidate.get("full_name", "Unknown")
        cache_key = candidate.get("cache_key", "")

        print(f"Processing: {full_name} (Score: {ai_score}, ID: {candidate_id})")

        # Skip if report already exists in Airtable for this rubric version
        if cache_key and airtable_has_report.get(cache_key):
            print(f"  Skipped (Airtable): report already exists")
            continue
        
        try:
            resume_text = ""

            # 1. Try local resume file first
            resume_path = get_resume_path(candidate)
            if resume_path:
                print(f"  Resume: {resume_path}")
                resume_text = extract_resume_text(resume_path) or ""

            # 2. Fall back to cv_text already stored in the CSV row
            if not resume_text:
                cv_text_csv = (candidate.get("cv_text") or "").strip()
                no_resume_marker = "no resume attached"
                if cv_text_csv and no_resume_marker not in cv_text_csv.lower():
                    resume_text = cv_text_csv
                    print(f"  Using cv_text from CSV (no local resume file)")

            # 3. Download fresh from resume_file URL (CSV or Airtable record)
            if not resume_text:
                resume_url = (candidate.get("resume_file") or "").strip()
                if not resume_url:
                    # Try to get URL from Airtable record
                    try:
                        at_fallback = AirtableClient()
                        formula = f"AND({{job_id}}={job_id}, {{candidate_id}}={candidate_id})"
                        recs = at_fallback.get_records_by_formula(formula)
                        if recs:
                            resume_url = (recs[0]["fields"].get("resume_file") or "").strip()
                    except Exception as _e:
                        print(f"  [WARN] Airtable URL lookup failed: {_e}")
                if resume_url:
                    print(f"  Downloading resume from URL...")
                    resume_text = _download_resume(resume_url, str(candidate_id))

            # 4. Last resort: cv_text directly from Airtable record
            if not resume_text:
                try:
                    at_fallback = AirtableClient()
                    formula = f"AND({{job_id}}={job_id}, {{candidate_id}}={candidate_id})"
                    recs = at_fallback.get_records_by_formula(formula)
                    if recs:
                        cv_airtable = (recs[0]["fields"].get("cv_text") or "").strip()
                        if cv_airtable and no_resume_marker not in cv_airtable.lower():
                            resume_text = cv_airtable
                            print(f"  Using cv_text from Airtable (fallback)")
                except Exception as _e:
                    print(f"  [WARN] Airtable cv_text fallback failed: {_e}")

            if not resume_text:
                print(f"  No resume text available, skipping")
                continue
            
            detailed_json = generate_detailed_json_with_ai(
                candidate, rubric, rubric_structure, resume_text, openai_client
            )
            
            json_path = Config.REPORTS_DIR / f"candidate_{candidate_id}_report.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(detailed_json, f, ensure_ascii=False, indent=2)
            
            html_content = generate_html_report(detailed_json)
            html_path = Config.REPORTS_DIR / f"candidate_{candidate_id}_report.html"
            with html_path.open("w", encoding="utf-8") as f:
                f.write(html_content)
            
            generated_count += 1
            print(f"  ✓ Generated: {json_path.name}, {html_path.name}")

            # Upload report to Airtable
            match_id = (candidate.get("match_id") or "").strip() or f"{job_id}-{candidate_id}"
            if update_airtable_report(
                match_id,
                int(candidate_id),
                int(job_id),
                detailed_json,
                html_path,
            ):
                uploaded_count += 1
                print(f"  ✓ Uploaded report to Airtable")
            else:
                print(f"  ✗ Airtable upload failed")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"AI-Powered Report Generation Complete")
    print(f"{'='*70}")
    print(f"Generated: {generated_count} reports (REAL AI scoring - no placeholders!)")
    print(f"Uploaded: {uploaded_count} to Airtable")
    print(f"Output: {Config.REPORTS_DIR}/")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
