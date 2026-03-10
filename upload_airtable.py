#!/usr/bin/env python3
"""upload_airtable.py

AIRTABLE UPLOAD — reads the scored CSV for a given JOB_ID and:
  1. Upserts candidate rows into the Airtable Candidate table
  2. Handles CV attachments (online mode: public URL; offline mode: local file upload)
  3. Attaches rubric text to each record

Usage:
  python upload_airtable.py <JOB_ID>

Environment variables (set in .env):
  AIRTABLE_TOKEN          - Airtable Personal Access Token (required)
  AIRTABLE_BASE_ID        - Base ID (default: app285aKVVr7JYL43)
  AIRTABLE_TABLE_ID       - Candidate table ID (default: tblJ2OkvaWI7vi0vI)
  INPUT_FILE              - Path to scored CSV (default: auto-derived from JOB_ID)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config
from airtable_client import AirtableClient
from utils import safe_filename

# =============================================================================
# CONFIG
# =============================================================================

if len(sys.argv) < 2:
    print("Usage: python upload_airtable.py <JOB_ID>")
    raise SystemExit(2)

JOB_ID = str(sys.argv[1]).strip()

INPUT_FILE = os.getenv(
    "INPUT_FILE",
    str(Config.get_scored_csv_path(JOB_ID))
)

UNIQUE_KEY_FIELD = Config.AIRTABLE_UNIQUE_KEY_FIELD   # "match_id"
CV_ATTACHMENT_FIELD = Config.AIRTABLE_CV_FIELD         # "CV"
RESUME_URL_FIELD = "resume_file"
RESUME_LOCAL_PATH_FIELD = "resume_local_path"

CREATE_MISSING = True
UPDATE_EXISTING = True


# =============================================================================
# INPUT LOADERS
# =============================================================================

def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_json(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() == ".csv":
        return load_csv(p)
    if p.suffix.lower() == ".json":
        return load_json(p)
    raise ValueError("INPUT_FILE must be .csv or .json")


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def is_http_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def build_cv_attachment(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Build CV attachment payload for public URLs (online mode)."""
    raw = row.get(RESUME_URL_FIELD)
    if raw is None:
        return None
    url = str(raw).strip()
    if not url or not is_http_url(url):
        return None
    name = str(row.get("full_name") or "candidate").strip() or "candidate"
    fn = f"{safe_filename(name)}_CV.pdf"
    return [{"url": url, "filename": fn}]


def normalize_value(field_name: str, v: Any) -> Any:
    """Normalize field values based on declared type in Config."""
    if v is None:
        return None

    if field_name in Config.TEXT_FIELDS:
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return str(v)

    if field_name in Config.NUMBER_FIELDS:
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s) if "." in s else int(s)
            except Exception:
                return None
        return None

    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def map_row_to_airtable_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map CSV/JSON row to Airtable field structure using Config.FIELD_MAP."""
    fields: Dict[str, Any] = {}
    for src, dst in Config.FIELD_MAP.items():
        if src in row:
            fields[dst] = normalize_value(dst, row.get(src))

    cv_attach = build_cv_attachment(row)
    if cv_attach:
        fields[CV_ATTACHMENT_FIELD] = cv_attach

    return fields


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    try:
        Config.validate()
    except ValueError as e:
        print(f"ERROR: {e}\n", file=sys.stderr)
        return 2

    try:
        airtable = AirtableClient()
    except ValueError as e:
        print(f"ERROR: {e}\n", file=sys.stderr)
        return 2

    print("Verifying Airtable credentials...")
    try:
        airtable.preflight_check()
        print("[OK] Connected to Airtable\n")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Loading input file: {INPUT_FILE}")
    try:
        rows = load_rows(INPUT_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not rows:
        print("No rows found in input file.")
        return 0
    print(f"[OK] Loaded {len(rows)} rows\n")

    # rubric_text is now a lookup field in Airtable (auto-populated via Job link)
    # — no longer written by the pipeline
    job_id_from_rows = str(rows[0].get("job_id") or JOB_ID).strip()

    # Resolve Job record ID for link field (once for all candidates)
    job_record_id: Optional[str] = airtable.get_job_record_id(job_id_from_rows)
    if job_record_id:
        print(f"[OK] Job record found: {job_record_id} — will link candidates\n")
    else:
        print(f"[WARN] No Job record found for job_id={job_id_from_rows} — candidates won't be linked\n")

    # Prepare records
    prepared: List[Dict[str, Any]] = []
    missing_key = 0

    for row in rows:
        key_val = row.get(UNIQUE_KEY_FIELD)
        if key_val is None or str(key_val).strip() == "":
            key_val = f"{row.get('job_id', '')}-{row.get('candidate_id', '')}"
        key_val = str(key_val).strip()
        if not key_val:
            missing_key += 1
            continue

        fields = map_row_to_airtable_fields(row)
        if job_record_id:
            fields["job"] = [job_record_id]

        prepared.append({
            "key": key_val,
            "fields": fields,
            "_resume_url": str(row.get(RESUME_URL_FIELD) or "").strip(),
            "_resume_local_path": str(row.get(RESUME_LOCAL_PATH_FIELD) or "").strip(),
            "_full_name": str(row.get("full_name") or "").strip(),
        })

    if missing_key:
        print(f"[WARN] Skipped {missing_key} rows missing '{UNIQUE_KEY_FIELD}'")

    if not prepared:
        print("Nothing to import after filtering.")
        return 0

    # Fetch existing records (full fields to detect cache_key changes)
    print("Fetching existing Airtable records...")
    existing_records_full = airtable.get_records_by_formula(
        f"{{job_id}}={job_id_from_rows}"
    )
    # match_id → record_id  (for upsert routing)
    existing: Dict[str, str] = {
        str(r["fields"].get(UNIQUE_KEY_FIELD) or "").strip(): r["id"]
        for r in existing_records_full
        if r["fields"].get(UNIQUE_KEY_FIELD)
    }
    # match_id → existing cache_key  (to detect rubric changes)
    existing_cache_keys: Dict[str, str] = {
        str(r["fields"].get(UNIQUE_KEY_FIELD) or "").strip(): str(r["fields"].get("cache_key") or "")
        for r in existing_records_full
        if r["fields"].get(UNIQUE_KEY_FIELD)
    }
    print(f"[OK] Found {len(existing)} existing records\n")

    # Classify: create vs update
    to_create: List[Dict[str, Any]] = []
    to_update: List[Dict[str, Any]] = []

    for item in prepared:
        key = item["key"]
        fields = item["fields"]
        if key in existing:
            if UPDATE_EXISTING:
                # If the rubric changed (new cache_key), clear the stale report
                # so Tier 2 regenerates it for this candidate.
                old_ck = existing_cache_keys.get(key, "")
                new_ck = str(fields.get("cache_key") or "")
                if old_ck and new_ck and old_ck != new_ck:
                    fields["ai_report_html"] = []
                    print(f"  [INFO] Rubric changed for {key} — clearing stale report")
                to_update.append({"id": existing[key], "fields": fields})
        else:
            if CREATE_MISSING:
                to_create.append(fields)

    print("Summary:")
    print(f"   - Records to create: {len(to_create)}")
    print(f"   - Records to update: {len(to_update)}\n")

    if to_create:
        print(f"Creating {len(to_create)} new records...")
        for batch in chunked(to_create, Config.AIRTABLE_BATCH_SIZE):
            airtable.batch_create(batch)
        print("[OK] Created\n")

    if to_update:
        print(f"Updating {len(to_update)} existing records...")
        for batch in chunked(to_update, Config.AIRTABLE_BATCH_SIZE):
            airtable.batch_update(batch)
        print("[OK] Updated\n")

    # Upload local CV files (offline mode)
    print("Processing CV attachments...")
    existing_after = airtable.get_all_records_by_key(UNIQUE_KEY_FIELD)

    uploaded = 0
    skipped_url = 0
    skipped_no_path = 0
    skipped_no_record = 0

    for item in prepared:
        key = item["key"]
        record_id = existing_after.get(key)
        if not record_id:
            skipped_no_record += 1
            continue

        resume_url = item.get("_resume_url") or ""
        resume_local_path = item.get("_resume_local_path") or ""
        full_name = item.get("_full_name") or ""

        if resume_url and is_http_url(resume_url):
            skipped_url += 1
            continue

        if not resume_local_path:
            skipped_no_path += 1
            continue

        preferred_fn = None
        if full_name:
            safe_name = safe_filename(full_name)
            if safe_name:
                preferred_fn = f"{safe_name}_CV{Path(resume_local_path).suffix or '.pdf'}"

        resp = airtable.upload_attachment_from_file(
            record_id,
            CV_ATTACHMENT_FIELD,
            resume_local_path,
            preferred_filename=preferred_fn,
        )
        if resp is not None:
            uploaded += 1

    print("[OK] CV Attachments processed:")
    if uploaded:
        print(f"   - Uploaded from local files: {uploaded}")
    if skipped_url:
        print(f"   - Used public URLs: {skipped_url}")
    if skipped_no_path:
        print(f"   - No CV file: {skipped_no_path}")
    if skipped_no_record:
        print(f"   - Skipped (no record ID): {skipped_no_record}")

    print("\n" + "=" * 70)
    print("[OK] Upload to Airtable completed successfully!")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
