#!/usr/bin/env python3
"""
generate_rubric.py — Generate AI-powered hiring rubrics from Manatal JDs.

Usage:
    python generate_rubric.py <job_id> /yaml
    python generate_rubric.py <job_id> /json

Environment Variables:
    MANATAL_API_KEY     (required) — Manatal Open API token
    OPENAI_API_KEY      (required) — OpenAI API key

Output:
    ./rubrics/rubric_<job_id>.yaml   (if /yaml)
    ./rubrics/rubric_<job_id>.json   (if /json)
"""

import sys
import os
import json
import re
import yaml
import datetime
import textwrap
import requests
from pathlib import Path
from html.parser import HTMLParser
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────
MANATAL_API_KEY = os.getenv("MANATAL_API_TOKEN", "")
MANATAL_BASE_URL = "https://api.manatal.com/open/v3"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
OUTPUT_DIR = Path("./rubrics")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── HTML STRIPPER ────────────────────────────────────────────────────────────
class HTMLStripper(HTMLParser):
    """Strip HTML tags and convert to clean text with basic structure."""

    def __init__(self):
        super().__init__()
        self.result = []
        self._in_li = False

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "br", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.result.append("\n")
        elif tag == "li":
            self.result.append("\n• ")
            self._in_li = True
        elif tag == "ul" or tag == "ol":
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.result.append("\n")
        elif tag == "li":
            self._in_li = False

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        raw = "".join(self.result)
        # Collapse multiple blank lines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def strip_html(html_text: str) -> str:
    """Convert HTML to clean plain text."""
    if not html_text:
        return ""
    stripper = HTMLStripper()
    stripper.feed(html_text)
    return stripper.get_text()


# ── MANATAL API ──────────────────────────────────────────────────────────────
def fetch_job_from_manatal(job_id: str) -> dict:
    """Fetch job details from Manatal Open API."""
    if not MANATAL_API_KEY:
        print("ERROR: MANATAL_API_KEY not set in .env or environment.")
        print("       Add it to .env:  MANATAL_API_KEY=your_key_here")
        sys.exit(1)

    url = f"{MANATAL_BASE_URL}/jobs/{job_id}/"
    headers = {
        "Authorization": f"Token {MANATAL_API_KEY}",
        "Content-Type": "application/json",
    }

    print(f"📡 Fetching job {job_id} from Manatal...")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ Found: {data.get('position_name', 'Unknown')} (ID: {data.get('id')})")
        return data
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            print(f"ERROR: Job {job_id} not found in Manatal.")
        elif resp.status_code == 401:
            print("ERROR: Manatal API key is invalid or expired.")
        else:
            print(f"ERROR: Manatal API returned {resp.status_code}: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not connect to Manatal API: {e}")
        sys.exit(1)


def prepare_jd_context(job_data: dict) -> str:
    """Build a rich JD context string from Manatal job data."""
    parts = []
    parts.append(f"Job ID: {job_data.get('id', 'N/A')}")
    parts.append(f"Position: {job_data.get('position_name', 'N/A')}")
    parts.append(f"Company/Organization ID: {job_data.get('organization', 'N/A')}")
    parts.append(f"Location: {job_data.get('city', '')}, {job_data.get('state', '')}, {job_data.get('country', '')}")
    parts.append(f"Contract: {job_data.get('contract_details', 'N/A')}")
    parts.append(f"Remote: {job_data.get('is_remote', 'N/A')}")

    salary_min = job_data.get("salary_min")
    salary_max = job_data.get("salary_max")
    currency = job_data.get("currency", "")
    if salary_min and salary_max:
        parts.append(f"Salary: {currency} {salary_min} - {salary_max}")

    industry = job_data.get("industry")
    if industry and isinstance(industry, dict):
        parts.append(f"Industry: {industry.get('name', 'N/A')}")

    desc_html = job_data.get("description", "")
    desc_text = strip_html(desc_html)
    parts.append(f"\n--- Full Job Description ---\n{desc_text}")

    return "\n".join(parts)


# ── PROMPT BUILDERS ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Technical Recruiting Rubric Generator creating rubrics with 
semantic ontology for AI-powered candidate scoring. You produce STRICTLY valid 
output in the requested format (YAML or JSON) with ZERO extra commentary.

# KNOWLEDGE BASE RULES

## Education Classification:
- "required"/"mandatory" → compliance
- "preferred" OR "equivalent experience accepted" → must_have (weight 5-10%)
- "plus"/"bonus" → nice_to_have (weight 1-2%)

## Weight Distribution:
- Must-Have total = 90%: Top skills 15-25%, Support 10-15%, Foundation 5-10%
- Nice-to-Have total = 10%: Advanced 2-3%, Certs 1-2%, Soft skills 0.5-1%

## Role Detection:
- Technical cues: engineer, developer, devops, ml, ai, data, cloud, python, java, etc.
- Sales cues: sales, account executive, quota, pipeline, crm, territory, etc.
- Corporate cues: hr, recruitment, finance, accounting, admin, coordinator, etc.
- Default to "technical" if mixed signals.

## Scoring Scale (fixed):
- 0: Not demonstrated
- 1: Mentioned only
- 2: Basic exposure
- 3: Hands-on experience
- 4: Advanced practical expertise
- 5: Expert level with leadership or architecture responsibility

## Seniority Inference from years:
- 0-2 years → junior
- 3-5 years → mid
- 6-10 years → senior
- 10+ years → lead or principal

## Compliance Gate:
- Work authorization: include ONLY if explicitly mentioned in JD
- Education: use STRICT_FIELD mode for technical roles (CS/SE/IT/related)
- Years of experience: derive minimum from JD text
""")


def build_yaml_prompt(jd_context: str, job_id: str) -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return textwrap.dedent(f"""\
Generate a complete, valid YAML rubric for the following job description.

**CRITICAL YAML FORMAT REQUIREMENTS:**
1. Output MUST be valid YAML — no markdown fences, no extra text
2. 2-space indentation (NO TABS)
3. Start with `rubric_name:` and end with the last metadata field
4. Must-have weights MUST total exactly 90
5. Nice-to-have weights MUST total exactly 10
6. Use these EXACT key names:
   - compliance items: `item`
   - must_have items: `requirement` + `weight`
   - nice_to_have items: `skill` + `weight`
7. All weights must be numbers, not strings
8. Today's date: {today}
9. Job ID: {job_id}

**REQUIRED SECTIONS (in order):**
- rubric_name (format: "PositionName_{job_id}_{today.replace('-','')}")
- role_applied
- generated_date
- version: "2.2"
- company
- jd_summary (max 100 words)
- normalized_terms (DETAILED semantic ontology — see examples below)
- compliance (list of dicts with 'item' key)
- must_have (list of dicts with 'requirement' and 'weight' keys, total=90)
- nice_to_have (list of dicts with 'skill' and 'weight' keys, total=10)
- scoring_rules (scale 0-5, pass_threshold 70, floor_rule)
- bias_guardrails
- metadata (rubric_type, seniority_level, domain, location)

# ============================================================
# CRITICAL: SEMANTIC ONTOLOGY (normalized_terms)
# ============================================================
# You MUST create a normalized_terms entry for EVERY technical skill,
# tool, platform, language, framework, methodology, and concept
# mentioned in the JD — in BOTH must_have AND nice_to_have sections.
#
# Typically this means 10-20+ entries. Do NOT skip any skill.
#
# Each entry MUST follow this EXACT structure (use underscores for keys):
#
#   Skill_Name:
#     aliases: ["variation1", "variation2", "acronym"]
#     parent_category: "broader_type"
#     child_specifics: ["sub_variant1", "sub_variant2"]
#     related_terms: ["ecosystem_tool1", "ecosystem_tool2"]
#     exclusions: ["commonly_confused_with"]
#     semantic_threshold: 0.88
#
# FIELD GUIDELINES:
# - aliases: ALL ways the skill might appear in a CV
#     Include: acronyms, full names, brand variations, common misspellings
#     e.g., a database might have aliases for its short name, CLI tool name, and full product name
# - parent_category: the broader skill type. Common categories:
#     programming_language, frontend_framework, backend_framework,
#     relational_database, nosql_database, cloud_platform, containerization,
#     ci_cd_tool, version_control, enterprise_platform, crm_platform,
#     erp_system, ai_ml_framework, data_platform, messaging_system,
#     monitoring_tool, testing_framework, methodology, certification,
#     business_process, financial_system, compliance_framework, etc.
# - child_specifics: specific versions, modules, sub-tools, or implementations
# - related_terms: tools/skills commonly used in the same ecosystem
# - exclusions: terms that look similar but are DIFFERENT technologies
# - semantic_threshold: how strict to match (0.85-0.95)
#     0.92-0.95: very specific/niche tools (strict match needed)
#     0.88: general frameworks and languages (standard)
#     0.85: broad categories and methodologies (loose match OK)
#
# COVERAGE — generate entries for ALL of these found in the JD:
#   - Programming languages
#   - Frontend and backend frameworks
#   - Databases and data stores
#   - Cloud platforms and services
#   - DevOps and CI/CD tools
#   - AI/ML frameworks and concepts
#   - APIs, protocols, and integration patterns
#   - Enterprise platforms (CRM, ERP, HRIS, etc.)
#   - Methodologies and frameworks (Agile, ITIL, etc.)
#   - Domain-specific tools (finance systems, HR tools, sales platforms, etc.)
#   - Certifications mentioned in the JD

**JOB DESCRIPTION:**
{jd_context}

Output ONLY valid YAML. No markdown code fences. No explanation. Start with rubric_name:
""")


def build_json_prompt(jd_context: str, job_id: str) -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return textwrap.dedent(f"""\
Generate a complete, valid JSON rubric for the following job description.

**CRITICAL JSON FORMAT REQUIREMENTS:**
1. Output MUST be valid JSON — no markdown fences, no extra text
2. Start with {{ and end with }}
3. Must-have weights MUST total exactly 90
4. Nice-to-have weights MUST total exactly 10
5. Today's date: {today}
6. Job ID: {job_id}
7. Education classification: If a bachelor's degree is REQUIRED/MANDATORY, put it ONLY in compliance_requirements (do NOT include it as a weighted must-have).
8. Do NOT include work authorization/visa/citizenship/location constraints unless explicitly stated in the JD text.
9. Education, Work Authorization, and Years of Experience should only treated as compliance_requirements if explicitly mentioned in the JD. Do NOT infer or assume these if not clearly stated.
10. Seniority level should be inferred from years of experience mentioned in the JD, if available. If no clear experience requirement is stated, default to "mid" level.
11. For technical roles, tag each must-have with importance_tier: "CRITICAL" | "IMPORTANT" | "FOUNDATIONAL".
12. Tier distribution constraint (technical roles): choose 2–3 CRITICAL, 2–4 IMPORTANT, and 0–2 FOUNDATIONAL. Do NOT put everything into IMPORTANT.


**REQUIRED JSON STRUCTURE:**
{{
  "job_id": "{job_id}",
  "role": "Full Job Title",
  "company": "Company Name",
  "seniority_level": "junior|mid|senior|lead",
  "rubric_name": "PositionName_YYYYMMDD",
  "rubric_version": "2.2",
  "generated_date": "{today}",
  "jd": {{
    "jd_summary": "One sentence summary (max 150 words)",
    "core_responsibilities": ["resp1", "resp2", ...],
    "must_haves_from_jd": ["req1", "req2", ...],
    "nice_to_haves_from_jd": ["skill1", "skill2", ...]
  }},
  "compliance_requirements": ["Requirement 1", "Requirement 2"],
  "scoring": {{
    "scale": {{"0":"Not demonstrated","1":"Mentioned only","2":"Basic exposure","3":"Hands-on experience","4":"Advanced practical expertise","5":"Expert level with leadership"}},
    "calculation": "Weighted average",
    "pass_threshold": 70,
    "floor_rule": "Any must-have < 2 triggers FAIL",
    "weighting": {{"must_have_total_weight_percent": 90, "nice_to_have_total_weight_percent": 10}}
  }},
  "requirements": {{
    "must_have": [
      {{
        "id": "MH1",
        "requirement": "Description",
        "importance_tier": "CRITICAL|IMPORTANT|FOUNDATIONAL",
        "weight": 0,
        "priority": "HIGHEST|HIGH|MEDIUM|LOW"
        "evidence_signals": ["signal1", "signal2", "signal3"],
        "negative_signals": ["red_flag1", "red_flag2"],
        "implementation_note": null
      }}
    ],
    "nice_to_have": [
      {{
        "id": "NH1",
        "skill": "Bonus skill description",
        "weight": 3
      }}
    ]
  }},
  "bias_guardrails": {{
    "protected_attributes": ["age","gender","race","ethnicity","religion","marital_status","disability"],
    "enforcement": "Strip protected attributes prior to scoring"
  }},
  "semantic_ontology": {{
    "normalized_terms": [ ... COMPREHENSIVE LIST ... ],
    "semantic_threshold_defaults": {{
      "highest_confidence": 0.9,
      "high_confidence": 0.88,
      "medium_confidence": 0.86,
      "min_acceptable": 0.85
    }}
  }},
  "report_format": {{
    "length_target": "1-2 pages",
    "output_language": "English",
    "sections": [
      "Role fit summary (overall score and hire/no-hire recommendation)",
      "Must-have requirements score breakdown (with evidence quotes/snippets from CV)",
      "Nice-to-have signals",
      "Gaps and risks (what is missing or unclear)",
      "Suggested interview questions to validate gaps",
      "Implementation notes / context (if any scoring relaxations apply)"
    ],
    "output_constraints": [
      "Do not consider or mention protected attributes.",
      "Every score claim must be supported by explicit CV evidence; otherwise mark as 'Not demonstrated'.",
      "Keep output within 1-2 pages; use bullet points and concise justification.",
      "If any must-have scores below 2, mark overall result as FAIL regardless of weighted score."
    ]
  }},
  "assumptions": ["assumption1", "assumption2"]
}}

# ============================================================
# CRITICAL: semantic_ontology.normalized_terms
# ============================================================
# This MUST be a COMPREHENSIVE flat array of ALL technical terms from the JD.
# Extract EVERY specific term actually mentioned or strongly implied in this JD.
#
# Target count:
#   - Technical roles: 30-50+ terms
#   - Non-technical roles (business, HR, finance): 15-30 terms
#
# WHAT TO INCLUDE (extract from THIS JD, do not invent):
#   - Programming languages mentioned
#   - Frontend and backend frameworks
#   - Databases, data stores, and query languages
#   - Cloud platforms and specific services
#   - DevOps, CI/CD, and infrastructure tools
#   - AI/ML frameworks, concepts, and techniques
#   - APIs, protocols, and integration patterns
#   - Enterprise platforms (CRM, ERP, HRIS, ATS, etc.)
#   - Specific vendor products and services named in the JD
#   - Methodologies and process frameworks (Agile, ITIL, MEDDPICC, etc.)
#   - Domain-specific concepts (e.g., "KYC", "AML" for compliance roles)
#   - Certifications mentioned
#   - Deployment platforms and hosting services
#   - Related ecosystem tools implied by the stack
#
# RULES:
# - Use the PRIMARY canonical name of each technology (e.g., the actual product name, not a generic category word)
# - Include specific products/vendor names when mentioned (e.g., the actual ATS, CRM, or cloud service)
# - Include domain concepts and methodologies, not just tools
# - Every term in must_have and nice_to_have requirements should be represented
# - Do NOT copy terms from other JDs — extract only from THIS job description

# ============================================================
# CRITICAL: importance_tier (must_have)
# ============================================================
# Add importance_tier to EVERY must_have item:
# - CRITICAL: explicitly emphasized in JD; core to success; repeated; tied to main deliverables
# - IMPORTANT: needed for solid performance; supporting core deliverables
# - FOUNDATIONAL: baseline expectations; lower differentiator
#
# For TECHNICAL roles: select 2–3 CRITICAL, 2–4 IMPORTANT, 0–2 FOUNDATIONAL.
# If unsure, make RAG + core stack (Python/backend or React/frontend) CRITICAL.

**AUTO-GENERATION RULES:**
- priority: weight >= 15 → "HIGHEST", 10-14 → "HIGH", 5-9 → "MEDIUM", < 5 → "LOW"
- evidence_signals: Generate 2-3 specific, detailed positive indicators per requirement
- negative_signals: Generate 1-2 specific anti-patterns per requirement
- implementation_note: null unless JD mentions flexibility/alternatives for that requirement
- seniority_level: Infer from years of experience in JD

**JOB DESCRIPTION:**
{jd_context}

Output ONLY valid JSON. No markdown code fences. No explanation. Start with {{
""")


# ── OPENAI API CALL ──────────────────────────────────────────────────────────
def call_llm(system: str, user_prompt: str) -> str:
    """Call OpenAI gpt-4o-mini and return the text response."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env or environment.")
        print("       Add to .env:  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "max_tokens": 8192,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    }

    print(f"🤖 Calling OpenAI ({OPENAI_MODEL}) to generate rubric...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            print("ERROR: OpenAI returned no choices")
            sys.exit(1)
        text = choices[0].get("message", {}).get("content", "").strip()

        # Log token usage
        usage = data.get("usage", {})
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)
        print(f"📊 Tokens: {input_tok} in + {output_tok} out = {input_tok + output_tok} total")

        return text

    except requests.exceptions.HTTPError as e:
        status = resp.status_code
        print(f"ERROR: OpenAI API returned HTTP {status}")
        try:
            err_body = resp.json()
            print(f"       {err_body.get('error', {}).get('message', str(e))}")
        except Exception:
            print(f"       {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not connect to OpenAI API: {e}")
        sys.exit(1)


# ── OUTPUT CLEANING ──────────────────────────────────────────────────────────
def clean_response(text: str, fmt: str) -> str:
    """Remove markdown fences and leading/trailing junk."""
    # Remove ```yaml or ```json fences
    text = re.sub(r"^```(?:yaml|json|yml)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    if fmt == "json":
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    return text


# ── VALIDATION ───────────────────────────────────────────────────────────────
def validate_json_rubric(content: str) -> tuple[bool, list[str]]:
    errors = []
    warnings = []

    # 1. Parse JSON safely
    try:
        rubric = json.loads(content)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON syntax: {e} (likely truncated output)"]

    if not isinstance(rubric, dict):
        return False, ["Root element must be a JSON object"]

    # 2. Required top-level fields
    REQUIRED_TOP = [
        "job_id", "role", "company", "seniority_level",
        "rubric_name", "rubric_version", "generated_date",
        "jd", "compliance_requirements", "scoring",
        "requirements", "bias_guardrails",
        "semantic_ontology", "report_format",
        "assumptions"
    ]

    for key in REQUIRED_TOP:
        if key not in rubric:
            errors.append(f"Missing required top-level field: '{key}'")

    # Stop if critical containers missing
    if errors:
        return False, errors

    # 3. Validate requirements
    reqs = rubric["requirements"]

    if "must_have" not in reqs or not isinstance(reqs["must_have"], list) or len(reqs["must_have"]) == 0:
        errors.append("'requirements.must_have' must be a non-empty list")

    if "nice_to_have" not in reqs or not isinstance(reqs["nice_to_have"], list):
        errors.append("'requirements.nice_to_have' must be a list")

    mh_total = 0
    nh_total = 0

    def expected_priority(weight):
        if weight >= 15:
            return "HIGHEST"
        if weight >= 10:
            return "HIGH"
        if weight >= 5:
            return "MEDIUM"
        return "LOW"

    # Validate must-have items
    for i, item in enumerate(reqs["must_have"]):
        for k in ["id", "requirement", "weight", "priority",
                  "evidence_signals", "negative_signals", "implementation_note"]:
            if k not in item:
                errors.append(f"must_have[{i}] missing '{k}'")

        weight = item.get("weight")
        if not isinstance(weight, (int, float)):
            errors.append(f"must_have[{i}].weight must be numeric")
        else:
            mh_total += weight
            expected = expected_priority(weight)
            if item.get("priority") != expected:
                errors.append(
                    f"must_have[{i}] priority mismatch: got '{item.get('priority')}', expected '{expected}'"
                )

        # Evidence signal count
        ev = item.get("evidence_signals", [])
        if not isinstance(ev, list) or not (2 <= len(ev) <= 3):
            errors.append(f"must_have[{i}] evidence_signals must contain 2–3 items")

        neg = item.get("negative_signals", [])
        if not isinstance(neg, list) or not (1 <= len(neg) <= 2):
            errors.append(f"must_have[{i}] negative_signals must contain 1–2 items")

    # Validate nice-to-have items
    for i, item in enumerate(reqs["nice_to_have"]):
        for k in ["id", "skill", "weight"]:
            if k not in item:
                errors.append(f"nice_to_have[{i}] missing '{k}'")

        weight = item.get("weight")
        if not isinstance(weight, (int, float)):
            errors.append(f"nice_to_have[{i}].weight must be numeric")
        else:
            nh_total += weight

    # 4. Enforce EXACT weight totals
    if mh_total != 90:
        errors.append(f"must_have total weight must equal 90 (got {mh_total})")

    if nh_total != 10:
        errors.append(f"nice_to_have total weight must equal 10 (got {nh_total})")

    # 5. Semantic ontology validation
    ont = rubric["semantic_ontology"]
    nt = ont.get("normalized_terms")

    if not isinstance(nt, list) or len(nt) < 15:
        errors.append("semantic_ontology.normalized_terms must be list with 15+ terms")

    th = ont.get("semantic_threshold_defaults")
    REQUIRED_THRESH = [
        "highest_confidence", "high_confidence",
        "medium_confidence", "min_acceptable"
    ]
    if not isinstance(th, dict):
        errors.append("semantic_threshold_defaults missing")
    else:
        for k in REQUIRED_THRESH:
            if k not in th:
                errors.append(f"semantic_threshold_defaults missing '{k}'")

    # 6. Report format validation
    rf = rubric["report_format"]
    for k in ["length_target", "output_language", "sections", "output_constraints"]:
        if k not in rf:
            errors.append(f"report_format missing '{k}'")

    if isinstance(rf.get("sections"), list) and len(rf["sections"]) < 5:
        errors.append("report_format.sections appears incomplete")

    if isinstance(rf.get("output_constraints"), list) and len(rf["output_constraints"]) < 3:
        errors.append("report_format.output_constraints appears incomplete")

    return len(errors) == 0, errors

# ── VALIDATION YAML ───────────────────────────────────────────────────────────────
def validate_yaml_rubric(content: str) -> tuple[bool, list[str]]:
    """Validate YAML rubric structure. Returns (is_valid, errors)."""
    errors = []
    warnings = []

    try:
        rubric = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML syntax: {e}"]

    if not isinstance(rubric, dict):
        return False, ["Root element must be a YAML mapping (dict)"]

    # Required sections
    for key in ["compliance", "must_have", "nice_to_have"]:
        if key not in rubric:
            errors.append(f"Missing required section: '{key}'")

    # compliance list of dicts with item
    comp = rubric.get("compliance")
    if comp is not None:
        if not isinstance(comp, list):
            errors.append("'compliance' must be a list")
        else:
            for i, item in enumerate(comp):
                if not isinstance(item, dict) or "item" not in item:
                    errors.append(f"compliance[{i}] must be a dict with 'item' key")

    # must_have list (non-empty) with requirement + weight
    mh = rubric.get("must_have")
    mh_total = 0
    if mh is not None:
        if not isinstance(mh, list) or len(mh) == 0:
            errors.append("'must_have' must be a non-empty list")
        else:
            for i, item in enumerate(mh):
                if not isinstance(item, dict):
                    errors.append(f"must_have[{i}] must be a dict")
                    continue
                if "requirement" not in item:
                    errors.append(f"must_have[{i}] missing 'requirement'")
                if "weight" not in item:
                    errors.append(f"must_have[{i}] missing 'weight'")
                else:
                    w = item["weight"]
                    if not isinstance(w, (int, float)):
                        errors.append(f"must_have[{i}].weight must be numeric")
                    else:
                        mh_total += w

    # nice_to_have list with skill + weight
    nh = rubric.get("nice_to_have")
    nh_total = 0
    if nh is not None:
        if not isinstance(nh, list):
            errors.append("'nice_to_have' must be a list")
        else:
            for i, item in enumerate(nh):
                if not isinstance(item, dict):
                    errors.append(f"nice_to_have[{i}] must be a dict")
                    continue
                if "skill" not in item:
                    errors.append(f"nice_to_have[{i}] missing 'skill'")
                if "weight" not in item:
                    errors.append(f"nice_to_have[{i}] missing 'weight'")
                else:
                    w = item["weight"]
                    if not isinstance(w, (int, float)):
                        errors.append(f"nice_to_have[{i}].weight must be numeric")
                    else:
                        nh_total += w

    # Enforce exact totals (since your prompts require exact 90/10)
    if mh_total != 90:
        errors.append(f"must_have total weight must equal 90 (got {mh_total})")
    if nh_total != 10:
        errors.append(f"nice_to_have total weight must equal 10 (got {nh_total})")

    # normalized_terms required for YAML (ontology)
    nt = rubric.get("normalized_terms")
    if not isinstance(nt, dict) or len(nt) < 5:
        errors.append("'normalized_terms' missing/too small — must be a dict with 5+ ontology entries")

    return len(errors) == 0, errors

# ── NORMALIZE RUBRIC ───────────────────────────────────────────────────────────────
def normalize_json_rubric(content: str) -> str:
    """
    Deterministically fix common LLM output issues:
    - priority must match weight buckets
    - must_have weights sum to 90
    - nice_to_have weights sum to 10 (optional)
    Returns normalized JSON string (pretty-printed).
    """
    try:
        rubric = json.loads(content)
    except Exception:
        return content  # validation will catch

    def expected_priority(w: float) -> str:
        if w >= 15:
            return "HIGHEST"
        if w >= 10:
            return "HIGH"
        if w >= 5:
            return "MEDIUM"
        return "LOW"

    reqs = rubric.get("requirements", {})
    mh = reqs.get("must_have", [])

    # Force must-have weights to follow your distribution rules
    allocate_must_have_weights_v2(mh)

# Recompute priorities to match weights
    for item in mh:
        apply_priority_from_weight(item)
        
        nh = reqs.get("nice_to_have", [])

    # 1) Fix priorities to match weights
    for item in mh:
        w = item.get("weight")
        if isinstance(w, (int, float)):
            item["priority"] = expected_priority(float(w))

    # 2) Force must_have sum to 90 (common failure is 100)
    mh_weights = [item.get("weight") for item in mh if isinstance(item, dict)]
    if all(isinstance(w, (int, float)) for w in mh_weights) and mh:
        mh_total = int(sum(mh_weights))
        if mh_total != 90:
            delta = mh_total - 90  # positive means we need to subtract
            if delta > 0:
                # subtract delta from lowest-priority items first, without going below 1
                # sort by weight ascending to minimally distort rubric
                mh_sorted = sorted(mh, key=lambda x: x.get("weight", 0))
                remaining = delta
                for item in mh_sorted:
                    w = item.get("weight", 0)
                    if not isinstance(w, (int, float)):
                        continue
                    # keep each must-have at least weight 1
                    reducible = max(0, int(w) - 1)
                    cut = min(reducible, remaining)
                    if cut > 0:
                        item["weight"] = int(w) - cut
                        remaining -= cut
                    if remaining == 0:
                        break
            else:
                # mh_total < 90, add missing weight to highest-weight item
                add = abs(delta)
                # choose current max weight item
                top = max(mh, key=lambda x: x.get("weight", 0))
                if isinstance(top.get("weight"), (int, float)):
                    top["weight"] = int(top["weight"]) + add

            # re-fix priorities after weight changes
            for item in mh:
                w = item.get("weight")
                if isinstance(w, (int, float)):
                    item["priority"] = expected_priority(float(w))

    # 3) (Optional) Force nice_to_have sum to 10
    nh_weights = [item.get("weight") for item in nh if isinstance(item, dict)]
    if all(isinstance(w, (int, float)) for w in nh_weights) and nh:
        nh_total = int(sum(nh_weights))
        if nh_total != 10:
            delta = nh_total - 10
            if delta > 0:
                nh_sorted = sorted(nh, key=lambda x: x.get("weight", 0))
                remaining = delta
                for item in nh_sorted:
                    w = item.get("weight", 0)
                    reducible = max(0, int(w) - 1)
                    cut = min(reducible, remaining)
                    if cut > 0:
                        item["weight"] = int(w) - cut
                        remaining -= cut
                    if remaining == 0:
                        break
            else:
                add = abs(delta)
                top = max(nh, key=lambda x: x.get("weight", 0))
                if isinstance(top.get("weight"), (int, float)):
                    top["weight"] = int(top["weight"]) + add

    return json.dumps(rubric, indent=2, ensure_ascii=False)

# ── RETRY LOGIC ──────────────────────────────────────────────────────────────
def generate_with_retry(system: str, prompt: str, fmt: str, max_retries: int = 2) -> str:
    """Generate rubric with validation and retry on failure."""
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"\n🔄 Retry attempt {attempt}/{max_retries}...")

        raw = call_llm(system, prompt)
        cleaned = clean_response(raw, fmt)
        if fmt == "json":
            cleaned = normalize_json_rubric(cleaned)

        print(f"\n🔍 Validating {fmt.upper()} structure...")

        if fmt == "yaml":
            valid, errs = validate_yaml_rubric(cleaned)
        else:
            valid, errs = validate_json_rubric(cleaned)

        if valid:
            print("✅ Validation PASSED")
            return cleaned
        else:
            print(f"❌ Validation FAILED ({len(errs)} error(s)):")
            for e in errs:
                print(f"   • {e}")

            if attempt < max_retries:
                # Append fix instructions to prompt for retry
                fix_msg = "\n\nPREVIOUS ATTEMPT HAD ERRORS:\n" + "\n".join(f"- {e}" for e in errs)
                fix_msg += "\n\nPlease fix ALL errors and output valid " + fmt.upper() + " only."
                prompt = prompt + fix_msg
    
    # Return best effort after retries exhausted
    print("\n⚠️  Returning best-effort output after retries. Manual review recommended.")
    return cleaned



def expected_priority(weight: float) -> str:
    if weight >= 15:
        return "HIGHEST"
    if weight >= 10:
        return "HIGH"
    if weight >= 5:
        return "MEDIUM"
    return "LOW"


def _distribute_with_bounds(total: int, n: int, min_w: int, max_w: int) -> list[int]:
    """
    Distribute integer total across n slots within [min_w, max_w].
    Deterministic: fills evenly, then fixes drift left-to-right.
    """
    if n <= 0:
        return []
    if n * min_w > total:
        # Can't satisfy minimums: relax by returning equal-ish and let caller handle drift
        base = max(1, total // n)
        weights = [base] * n
        weights[0] += total - sum(weights)
        return weights

    if n * max_w < total:
        # Can't satisfy maximums: cap all, then caller handles remaining drift elsewhere
        weights = [max_w] * n
        return weights

    base = total // n
    rem = total - base * n
    weights = [base + (1 if i < rem else 0) for i in range(n)]

    # Clamp
    weights = [min(max_w, max(min_w, w)) for w in weights]

    # Repair drift after clamping
    drift = total - sum(weights)
    i = 0
    while drift != 0 and n > 0:
        if drift > 0 and weights[i] < max_w:
            weights[i] += 1
            drift -= 1
        elif drift < 0 and weights[i] > min_w:
            weights[i] -= 1
            drift += 1
        i = (i + 1) % n

    return weights


def allocate_must_have_weights_v2(must_haves: list[dict]) -> None:
    """
    Production-grade allocator:
    - uses importance_tier to infer relative importance
    - assigns weights summing to EXACTLY 90
    - respects min/max bounds per tier
    - recomputes priority from weights
    Mutates must_haves in-place.
    """
    if not must_haves:
        return

    # Normalize tiers (default IMPORTANT)
    for it in must_haves:
        tier = (it.get("importance_tier") or "IMPORTANT").upper()
        if tier not in ("CRITICAL", "IMPORTANT", "FOUNDATIONAL"):
            tier = "IMPORTANT"
        it["importance_tier"] = tier

    # If no CRITICAL present, promote the first item to CRITICAL for sane scoring
    if not any(it["importance_tier"] == "CRITICAL" for it in must_haves):
        must_haves[0]["importance_tier"] = "CRITICAL"

    # Buckets (preserve original order later)
    critical = [it for it in must_haves if it["importance_tier"] == "CRITICAL"]
    important = [it for it in must_haves if it["importance_tier"] == "IMPORTANT"]
    foundational = [it for it in must_haves if it["importance_tier"] == "FOUNDATIONAL"]

    nC, nI, nF = len(critical), len(important), len(foundational)
    N = len(must_haves)

    TOTAL = 90

    # Tier bounds (you can tune these)
    bounds = {
        "CRITICAL": (15, 25),
        "IMPORTANT": (10, 15),
        "FOUNDATIONAL": (5, 10),
    }

    # --- Dynamic tier budgets (v2) ---
    #
    # Goals:
    # - If there are more CRITICAL items, allocate more total budget to CRITICAL.
    # - Still keep IMPORTANT meaningful.
    # - FOUNDATIONAL should be small but non-zero if present.
    #
    # Approach:
    # - Allocate FOUNDATIONAL first if present: between 8 and 15, scaling with count.
    # - Allocate CRITICAL next: between 35 and 55, scaling with share of critical items.
    # - IMPORTANT gets the remainder.
    #
    # Then re-balance if any tier violates feasibility (min/max total possible).

    # Foundational budget
    if nF == 0:
        budget_F = 0
    else:
        # 8..15 depending on count, but not exceeding max possible
        budget_F = min(15, max(8, 5 * nF))  # 1 item -> 8, 2 items -> 10, 3 items -> 15
        # ensure within feasible range
        minF = nF * bounds["FOUNDATIONAL"][0]
        maxF = nF * bounds["FOUNDATIONAL"][1]
        budget_F = min(maxF, max(minF, budget_F))

    remaining_after_F = TOTAL - budget_F

    # Critical budget (dynamic with cap)
    shareC = nC / max(1, N)
    # map shareC into 35..55 (more critical items => more critical budget)
    budget_C = int(round(35 + (55 - 35) * shareC))
    budget_C = min(55, max(35, budget_C))

    # ensure feasible for nC
    minC = nC * bounds["CRITICAL"][0]
    maxC = nC * bounds["CRITICAL"][1]
    if nC == 0:
        budget_C = 0
    else:
        budget_C = min(maxC, max(minC, budget_C))

    # Important budget = remainder
    budget_I = remaining_after_F - budget_C

    # ensure feasible for nI
    minI = nI * bounds["IMPORTANT"][0]
    maxI = nI * bounds["IMPORTANT"][1]
    if nI == 0:
        budget_I = 0
    else:
        # If remainder is outside feasible range, clamp and push/pull from critical first
        if budget_I < minI:
            need = minI - budget_I
            # try take from critical if possible
            takeC = 0
            if nC > 0:
                takeC = min(need, budget_C - minC)
                budget_C -= takeC
                need -= takeC
            # then from foundational if possible
            takeF = 0
            if need > 0 and nF > 0:
                minF = nF * bounds["FOUNDATIONAL"][0]
                takeF = min(need, budget_F - minF)
                budget_F -= takeF
                need -= takeF
            budget_I = minI  # set to min feasible
        elif budget_I > maxI:
            extra = budget_I - maxI
            # push extra into critical if room, else foundational
            pushC = 0
            if nC > 0:
                pushC = min(extra, maxC - budget_C)
                budget_C += pushC
                extra -= pushC
            pushF = 0
            if extra > 0 and nF > 0:
                maxF = nF * bounds["FOUNDATIONAL"][1]
                pushF = min(extra, maxF - budget_F)
                budget_F += pushF
                extra -= pushF
            budget_I = maxI

    # Final drift fix to guarantee TOTAL
    drift = TOTAL - (budget_C + budget_I + budget_F)
    # Prefer adjusting IMPORTANT, then CRITICAL, then FOUNDATIONAL within feasibility
    def _adjust_budget(tier: str, delta: int) -> int:
        nonlocal budget_C, budget_I, budget_F
        if tier == "IMPORTANT":
            n, (mn, mx) = nI, bounds["IMPORTANT"]
            if n == 0:
                return delta
            minT, maxT = n * mn, n * mx
            new = max(minT, min(maxT, budget_I + delta))
            delta_left = delta - (new - budget_I)
            budget_I = new
            return delta_left
        if tier == "CRITICAL":
            n, (mn, mx) = nC, bounds["CRITICAL"]
            if n == 0:
                return delta
            minT, maxT = n * mn, n * mx
            new = max(minT, min(maxT, budget_C + delta))
            delta_left = delta - (new - budget_C)
            budget_C = new
            return delta_left
        if tier == "FOUNDATIONAL":
            n, (mn, mx) = nF, bounds["FOUNDATIONAL"]
            if n == 0:
                return delta
            minT, maxT = n * mn, n * mx
            new = max(minT, min(maxT, budget_F + delta))
            delta_left = delta - (new - budget_F)
            budget_F = new
            return delta_left
        return delta

    if drift != 0:
        drift = _adjust_budget("IMPORTANT", drift)
    if drift != 0:
        drift = _adjust_budget("CRITICAL", drift)
    if drift != 0:
        drift = _adjust_budget("FOUNDATIONAL", drift)
    # If still drifting (rare due to feasibility), put remainder into the first item
    # by later item-level drift fix.

    # Distribute within tiers
    wC = _distribute_with_bounds(budget_C, nC, *bounds["CRITICAL"])
    wI = _distribute_with_bounds(budget_I, nI, *bounds["IMPORTANT"])
    wF = _distribute_with_bounds(budget_F, nF, *bounds["FOUNDATIONAL"])

    itC = iter(wC)
    itI = iter(wI)
    itF = iter(wF)

    # Assign weights back in original order
    for it in must_haves:
        tier = it["importance_tier"]
        if tier == "CRITICAL":
            it["weight"] = int(next(itC))
        elif tier == "FOUNDATIONAL":
            it["weight"] = int(next(itF))
        else:
            it["weight"] = int(next(itI))

    # Item-level drift fix to guarantee sum EXACTLY 90
    total_now = sum(int(it.get("weight", 0)) for it in must_haves)
    item_drift = TOTAL - total_now
    if item_drift != 0:
        # apply drift to first CRITICAL item if possible, else first item
        target = next((it for it in must_haves if it["importance_tier"] == "CRITICAL"), must_haves[0])
        target["weight"] = int(target["weight"]) + item_drift

    # Recompute priorities from final weights
    for it in must_haves:
        it["priority"] = expected_priority(float(it["weight"]))

# ── ALLOCATE MUST HAVE WEIGHTS ─────────────────────────────────────────────────────────────
def allocate_must_have_weights(must_haves: list[dict]) -> None:
    """
    Deterministically assign weights to must_have items to:
      - sum to 90
      - follow: top skills 15–25, support 10–15, foundation 5–10
    Mutates must_haves in-place.
    """
    if not must_haves:
        return

    n = len(must_haves)

    # Choose a simple, robust scheme based on count
    # Aim: 2 top items + remaining support/foundation.
    if n == 1:
        weights = [90]
    elif n == 2:
        weights = [45, 45]
    elif n == 3:
        weights = [25, 20, 45]  # not ideal but rare; consider splitting requirements instead
    elif n == 4:
        weights = [25, 20, 25, 20]  # sums 90
    elif n == 5:
        weights = [25, 20, 15, 15, 15]  # sums 90 (good default)
    elif n == 6:
        weights = [20, 15, 15, 15, 15, 10]  # sums 90
    elif n == 7:
        weights = [18, 15, 14, 13, 12, 10, 8]  # sums 90
    elif n == 8:
        weights = [18, 15, 12, 12, 11, 9, 7, 6]  # sums 90
    else:
        # For 9+ items, assign diminishing weights then normalize to 90
        base = [15, 12, 10, 10, 9, 8, 7, 6, 5]
        weights = (base + [4] * (n - len(base)))[:n]
        total = sum(weights)
        # scale to 90 while keeping ints
        scaled = [max(1, round(w * 90 / total)) for w in weights]
        # fix rounding drift
        drift = 90 - sum(scaled)
        scaled[0] += drift
        weights = scaled

    # Assign to items (keep original ordering)
    for item, w in zip(must_haves, weights):
        item["weight"] = int(w)

# ── Recompute priority after weights are set──────────────────────────────────────────────────────────────
def apply_priority_from_weight(item: dict) -> None:
    w = item.get("weight")
    if not isinstance(w, (int, float)):
        return
    w = float(w)
    if w >= 15:
        item["priority"] = "HIGHEST"
    elif w >= 10:
        item["priority"] = "HIGH"
    elif w >= 5:
        item["priority"] = "MEDIUM"
    else:
        item["priority"] = "LOW"

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_rubric.py <job_id>")
        print("Example: python generate_rubric.py 3419430")
        sys.exit(1)

    job_id = sys.argv[1].strip()

    print("=" * 70)
    print(f"  RUBRIC GENERATOR — Job {job_id}")
    print("=" * 70)
    print()

    from airtable_client import AirtableClient
    at = AirtableClient()

    # ── 1. Fetch JD from Manatal (needed for word-count check) ────────────
    job_data = fetch_job_from_manatal(job_id)
    jd_context = prepare_jd_context(job_data)
    jd_word_count = len(jd_context.split())
    print(f"📄 JD extracted ({len(jd_context)} chars, {jd_word_count} words)")

    # ── 2. JD change detection via word count ─────────────────────────────
    # Compare current JD word count against what is stored in the Job table.
    # If different → JD has changed → delete stale rubric so it gets regenerated.
    job_record_id = at.get_job_record_id(job_id)
    if job_record_id:
        from airtable_client import Config as _Cfg
        job_client = AirtableClient(
            token=at.token, base_id=at.base_id, table_id=_Cfg.AIRTABLE_JOB_TABLE_ID
        )
        job_records = job_client.get_records_by_formula(f"{{job_id}}={job_id}")
        stored_word_cnt = job_records[0]["fields"].get("word_cnt") if job_records else None

        if stored_word_cnt is not None and int(stored_word_cnt) != jd_word_count:
            print(
                f"⚠️  JD change detected for job_id={job_id}: "
                f"stored word_cnt={stored_word_cnt}, current={jd_word_count}"
            )
            print("   Updating word_cnt and deleting stale rubric...")
            job_client.update_record(job_record_id, {"word_cnt": jd_word_count})
            deleted = at.delete_rubric(job_id)
            if deleted:
                print("   🗑️  Stale rubric deleted — will regenerate.")
            else:
                print("   No existing rubric to delete.")
        elif stored_word_cnt is not None:
            print(f"✅ JD unchanged (word_cnt={jd_word_count}) — rubric is up to date.")
        else:
            # Job record exists but word_cnt not yet set — write it now
            job_client.update_record(job_record_id, {"word_cnt": jd_word_count})
            print(f"📝 word_cnt initialised to {jd_word_count} for job_id={job_id}")
    else:
        print(f"ℹ️  No Job record found for job_id={job_id} — skipping word count check.")

    # ── 3. Check Airtable for existing rubric ─────────────────────────────
    print(f"\nChecking Airtable for existing rubric (job_id={job_id})...")
    existing = at.get_rubric(job_id)
    if existing:
        print(f"✅ Rubric already exists in Airtable for job_id={job_id}. Skipping generation.")
        _print_rubric_summary(existing)
        return 0  # exit 0 = rubric unchanged, no rescore needed

    print("No rubric found in Airtable. Generating now...\n")

    # ── 4. Build prompt and generate rubric ───────────────────────────────
    prompt = build_json_prompt(jd_context, job_id)
    content = generate_with_retry(SYSTEM_PROMPT, prompt, "json", max_retries=2)
    rubric = json.loads(content)

    # ── 5. Upsert Job record (creates it if missing, updates word_cnt) ────
    org = job_data.get("organization")
    client_id   = int(org) if isinstance(org, (int, float)) else None
    client_name = job_data.get("organisation_name", "")
    at.upsert_job(
        job_id=job_id,
        job_name=job_data.get("position_name", ""),
        jd_text=jd_context[:50_000],
        client_id=client_id,
        client_name=client_name,
        word_cnt=jd_word_count,
    )
    print(f"✅ Job record upserted in Airtable for job_id={job_id} (word_cnt={jd_word_count})")

    # ── 6. Upload rubric ───────────────────────────────────────────────────
    at.upsert_rubric(job_id, rubric)

    print()
    print("=" * 70)
    print(f"  ✅ RUBRIC UPLOADED TO AIRTABLE for job_id={job_id}")
    print("=" * 70)

    _print_rubric_summary(rubric)
    print()
    return 2  # exit 2 = rubric was (re)generated → trigger rescore of existing candidates


def _print_rubric_summary(rubric: dict) -> None:
    """Print a brief summary of a rubric dict."""
    try:
        reqs = rubric.get("requirements", {})
        mh = reqs.get("must_have", [])
        nh = reqs.get("nice_to_have", [])
        comp = rubric.get("compliance_requirements", [])
        mh_w = sum(r.get("weight", 0) for r in mh)
        nh_w = sum(r.get("weight", 0) for r in nh)
        ont = rubric.get("semantic_ontology", {}).get("normalized_terms", [])
        print(f"\n📋 Summary:")
        print(f"   Role:             {rubric.get('role', 'N/A')}")
        print(f"   Seniority:        {rubric.get('seniority_level', 'N/A')}")
        print(f"   Compliance items: {len(comp)}")
        print(f"   Must-have:        {len(mh)} requirements (total weight: {mh_w}%)")
        print(f"   Nice-to-have:     {len(nh)} skills (total weight: {nh_w}%)")
        print(f"   Ontology terms:   {len(ont)}")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
