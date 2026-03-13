#!/usr/bin/env python3
"""airtable_client.py

Airtable API client wrapper for the recruitment pipeline.
Provides a clean interface for common Airtable operations.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config import Config


class AirtableClient:
    """Simple Airtable API client for recruitment pipeline operations."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_id: Optional[str] = None,
        table_id: Optional[str] = None,
    ):
        self.token = (token or Config.AIRTABLE_TOKEN).strip()
        self.base_id = (base_id or Config.AIRTABLE_BASE_ID).strip()
        self.table_id = (table_id or Config.AIRTABLE_TABLE_ID).strip()

        if not self.token:
            raise ValueError(
                "AIRTABLE_TOKEN not set. "
                "Set it in .env file or pass to AirtableClient(token=...)"
            )

        if not self.base_id or not self.table_id:
            raise ValueError("AIRTABLE_BASE_ID and AIRTABLE_CANDIDATE_TABLE_ID must be set")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"https://api.airtable.com/v0/{self.base_id}/{path}"

    def preflight_check(self) -> bool:
        """Verify connection to Airtable. Raises RuntimeError on failure."""
        url = self._url(f"{self.table_id}?pageSize=1")
        r = requests.get(url, headers=self._headers(), timeout=30)

        if r.status_code == 200:
            return True

        raise RuntimeError(
            f"Airtable preflight check failed: {r.status_code}\n"
            f"Response: {r.text}\n"
            f"Check your AIRTABLE_TOKEN, AIRTABLE_BASE_ID, and AIRTABLE_CANDIDATE_TABLE_ID"
        )

    def get_all_records_by_key(self, key_field: str) -> Dict[str, str]:
        """Get all existing records indexed by a unique key field.

        Returns a dict mapping key values → Airtable record IDs.
        """
        out: Dict[str, str] = {}
        offset: Optional[str] = None

        while True:
            params: Dict[str, Any] = {"pageSize": 100}
            if offset:
                params["offset"] = offset

            r = requests.get(
                self._url(self.table_id),
                headers=self._headers(),
                params=params,
                timeout=60,
            )

            if not r.ok:
                raise RuntimeError(f"Airtable GET failed: {r.status_code}\n{r.text}")

            data = r.json()
            for rec in data.get("records", []):
                fields = rec.get("fields", {}) or {}
                key_val = fields.get(key_field)
                if key_val is None:
                    continue
                key_str = str(key_val).strip()
                if key_str:
                    out[key_str] = rec["id"]

            offset = data.get("offset")
            if not offset:
                break

        return out

    def get_records_by_formula(self, formula: str) -> List[Dict[str, Any]]:
        """Fetch all records matching a filterByFormula."""
        out: List[Dict[str, Any]] = []
        offset: Optional[str] = None

        while True:
            params: Dict[str, Any] = {"pageSize": 100, "filterByFormula": formula}
            if offset:
                params["offset"] = offset

            r = requests.get(
                self._url(self.table_id),
                headers=self._headers(),
                params=params,
                timeout=60,
            )

            if not r.ok:
                raise RuntimeError(f"Airtable GET failed: {r.status_code}\n{r.text}")

            data = r.json()
            for rec in data.get("records", []):
                out.append({"id": rec["id"], "fields": rec.get("fields") or {}})

            offset = data.get("offset")
            if not offset:
                break

        return out

    def find_record_by_field(
        self,
        field: str,
        value: str,
        max_records: int = 1,
    ) -> Optional[str]:
        """Find a record by field value. Returns the record ID or None."""
        params = {
            "filterByFormula": f"{{{field}}}='{value}'",
            "maxRecords": max_records,
        }

        r = requests.get(
            self._url(self.table_id),
            headers=self._headers(),
            params=params,
            timeout=30,
        )

        if not r.ok:
            print(f"[WARN] Airtable lookup failed: {r.status_code}")
            return None

        records = r.json().get("records", [])
        return records[0]["id"] if records else None

    def batch_create(self, records_fields: List[Dict[str, Any]]) -> None:
        """Create multiple records in batch (max 10 per call)."""
        payload = {"records": [{"fields": rf} for rf in records_fields]}
        r = requests.post(
            self._url(self.table_id),
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )

        if not r.ok:
            raise RuntimeError(f"Airtable CREATE failed: {r.status_code}\n{r.text}")

    def batch_update(self, records: List[Dict[str, Any]]) -> None:
        """Update multiple records in batch (max 10 per call).

        Args:
            records: List of dicts with "id" and "fields" keys.
        """
        payload = {"records": records}
        r = requests.patch(
            self._url(self.table_id),
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )

        if not r.ok:
            raise RuntimeError(f"Airtable UPDATE failed: {r.status_code}\n{r.text}")

    def update_record(self, record_id: str, fields: Dict[str, Any]) -> bool:
        """Update a single record. Returns True on success."""
        url = self._url(f"{self.table_id}/{record_id}")
        payload = {"fields": fields}

        r = requests.patch(
            url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )

        if not r.ok:
            print(f"[WARN] Record update failed: {r.status_code}\n{r.text}")

        return r.ok

    def upload_attachment_from_bytes(
        self,
        record_id: str,
        field_name: str,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Upload a file attachment from bytes via Airtable's uploadAttachment API."""
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "application/octet-stream"

        url = f"https://content.airtable.com/v0/{self.base_id}/{record_id}/{field_name}/uploadAttachment"
        b64 = base64.b64encode(file_content).decode("ascii")

        payload = {
            "contentType": content_type,
            "filename": filename,
            "file": b64,
        }

        r = requests.post(
            url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=60,
        )

        if not r.ok:
            print(f"[WARN] Attachment upload failed: {r.status_code}\n{r.text}")
            return None

        return r.json()

    def upload_attachment_from_file(
        self,
        record_id: str,
        field_name: str,
        file_path: str | Path,
        preferred_filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Upload a local file as an attachment to an Airtable record."""
        p = Path(file_path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()

        if not p.exists() or not p.is_file():
            print(f"[WARN] File not found: {p}")
            return None

        size = p.stat().st_size
        if size > Config.AIRTABLE_UPLOAD_MAX_BYTES:
            print(
                f"[WARN] File too large for direct upload "
                f"(>{Config.AIRTABLE_UPLOAD_MAX_BYTES} bytes): {p} ({size} bytes)"
            )
            return None

        content_type, _ = mimetypes.guess_type(str(p))
        if not content_type:
            content_type = "application/pdf" if p.suffix.lower() == ".pdf" else "application/octet-stream"

        filename = preferred_filename or p.name
        return self.upload_attachment_from_bytes(
            record_id=record_id,
            field_name=field_name,
            file_content=p.read_bytes(),
            filename=filename,
            content_type=content_type,
        )

    def upload_text_as_attachment(
        self,
        record_id: str,
        field_name: str,
        text_content: str,
        filename: str,
    ) -> Optional[Dict[str, Any]]:
        """Upload text content (HTML, JSON, etc.) as a file attachment."""
        ext = Path(filename).suffix.lower()
        content_type_map = {
            ".json": "application/json",
            ".html": "text/html",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
        }
        content_type = content_type_map.get(ext, "text/plain")

        return self.upload_attachment_from_bytes(
            record_id=record_id,
            field_name=field_name,
            file_content=text_content.encode("utf-8"),
            filename=filename,
            content_type=content_type,
        )

    # ------------------------------------------------------------------
    # Job helpers (Job table)
    # ------------------------------------------------------------------

    def get_job_record_id(self, job_id: str) -> Optional[str]:
        """Return the Airtable record ID for a Job row, or None if not found."""
        client = AirtableClient(
            token=self.token,
            base_id=self.base_id,
            table_id=Config.AIRTABLE_JOB_TABLE_ID,
        )
        # job_id is the numeric primary field in the Job table
        records = client.get_records_by_formula(f"{{job_id}}={job_id}")
        return records[0]["id"] if records else None

    def upsert_job(
        self,
        job_id: str,
        job_name: str,
        jd_text: str = "",
        client_id: Optional[int] = None,
        client_name: str = "",
        word_cnt: Optional[int] = None,
    ) -> str:
        """Create or update a Job record. Returns the Airtable record ID."""
        client = AirtableClient(
            token=self.token,
            base_id=self.base_id,
            table_id=Config.AIRTABLE_JOB_TABLE_ID,
        )
        records = client.get_records_by_formula(f"{{job_id}}={job_id}")
        existing_id = records[0]["id"] if records else None

        payload: Dict[str, Any] = {
            "job_id":   int(job_id),
            "job_name": job_name,
        }
        if jd_text:
            payload["jd"] = jd_text
        if client_id is not None:
            payload["client_id"] = int(client_id)
        if client_name:
            payload["client_name"] = client_name
        if word_cnt is not None:
            payload["word_cnt"] = int(word_cnt)

        if existing_id:
            client.update_record(existing_id, payload)
            return existing_id
        else:
            # batch_create returns nothing; re-fetch to get the new record ID
            client.batch_create([payload])
            records = client.get_records_by_formula(f"{{job_id}}={job_id}")
            return records[0]["id"] if records else ""

    # ------------------------------------------------------------------
    # Rubric helpers (stored directly on the Job table)
    # ------------------------------------------------------------------

    def get_rubric(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the rubric JSON for a job from the Job table.

        Looks up by the singleLineText job_id field.
        """
        client = AirtableClient(
            token=self.token,
            base_id=self.base_id,
            table_id=Config.AIRTABLE_JOB_TABLE_ID,
        )
        # job_id is singleLineText in Job table — needs quotes in formula
        records = client.get_records_by_formula(f'{{job_id}}="{job_id}"')
        if not records:
            return None
        raw = records[0]["fields"].get("rubric_json", "")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def delete_rubric(self, job_id: str) -> bool:
        """Clear the rubric fields on the Job record for a given job_id.

        Returns True if a Job record was found and cleared, False otherwise.
        """
        client = AirtableClient(
            token=self.token,
            base_id=self.base_id,
            table_id=Config.AIRTABLE_JOB_TABLE_ID,
        )
        records = client.get_records_by_formula(f'{{job_id}}="{job_id}"')
        if not records:
            return False
        record_id = records[0]["id"]
        client.update_record(record_id, {"rubric_json": "", "rubric_name": ""})
        return True

    def upsert_rubric(self, job_id: str, rubric: Dict[str, Any]) -> None:
        """Write rubric_json and rubric_name directly onto the Job record.

        rubric_name is set to RoleName_YYYYMMDD (e.g. SeniorMSDynamicsCRMDeveloper_20260309).
        If no Job record exists for this job_id, a minimal one is created.
        """
        from datetime import date

        client = AirtableClient(
            token=self.token,
            base_id=self.base_id,
            table_id=Config.AIRTABLE_JOB_TABLE_ID,
        )

        # Build rubric_name from role + today's date
        role_raw = rubric.get("role", "Unknown")
        role_slug = "".join(w.capitalize() for w in role_raw.split())
        rubric_name = f"{role_slug}_{date.today().strftime('%Y%m%d')}"

        payload: Dict[str, Any] = {
            "rubric_name": rubric_name,
            "rubric_json": json.dumps(rubric, ensure_ascii=False),
        }

        # Find existing Job record by singleLineText job_id field
        records = client.get_records_by_formula(f'{{job_id}}="{job_id}"')
        if records:
            client.update_record(records[0]["id"], payload)
        else:
            # No Job record yet — create a minimal one
            payload["job_id"] = str(job_id)
            client.batch_create([payload])
