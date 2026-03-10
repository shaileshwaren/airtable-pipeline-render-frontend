#!/usr/bin/env python3
"""config.py

Centralized configuration for the Airtable recruitment pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Global configuration settings."""

    # =========================
    # Paths
    # =========================
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = Path(os.getenv("EXPORT_PATH", str(BASE_DIR / "output")))
    RUBRIC_DIR = Path(os.getenv("RUBRIC_DIR", str(BASE_DIR / "rubrics")))
    OFFLINE_INPUT_DIR = BASE_DIR / "offline_input"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    UPLOAD_DIR = OUTPUT_DIR / "upload"

    CACHE_FILE = Path(os.getenv("CACHE_FILE", str(OUTPUT_DIR / "scored_cache.json")))

    # =========================
    # API Keys & Tokens
    # =========================
    MANATAL_API_TOKEN = os.getenv("MANATAL_API_TOKEN", "").strip()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN", "").strip()

    # =========================
    # Airtable Configuration
    # =========================
    AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "app285aKVVr7JYL43").strip()
    AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE_ID", "tblJ2OkvaWI7vi0vI").strip()
    AIRTABLE_RUBRIC_TABLE_ID = os.getenv("AIRTABLE_RUBRIC_TABLE_ID", "tblZgr3F6DWEOorG7").strip()
    AIRTABLE_JOB_TABLE_ID = os.getenv("AIRTABLE_JOB_TABLE_ID", "tblCV6w4fGex9VgzK").strip()

    # Airtable field names
    AIRTABLE_CV_FIELD = "CV"
    AIRTABLE_UNIQUE_KEY_FIELD = "match_id"

    # Airtable limits
    AIRTABLE_BATCH_SIZE = 10
    AIRTABLE_UPLOAD_MAX_BYTES = 5 * 1024 * 1024  # 5 MB limit for direct uploads

    # =========================
    # OpenAI Configuration
    # =========================
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    REPORT_OPENAI_MODEL = os.getenv("REPORT_OPENAI_MODEL", "gpt-4o")

    # =========================
    # Manatal API Configuration
    # =========================
    MANATAL_BASE_URL = "https://api.manatal.com/open/v3"
    MANATAL_PAGE_SIZE = 100

    # =========================
    # Pipeline Settings
    # =========================
    TARGET_STAGE_NAME  = os.getenv("TARGET_STAGE_NAME",  "New Candidates")
    TARGET_STAGE_AFTER = os.getenv("TARGET_STAGE_AFTER", "AI Screened")

    DOWNLOAD_RESUMES = True
    SKIP_ALREADY_SCORED = True
    FORCE_RESCORE = False

    CV_EXTENSIONS = {".pdf", ".docx", ".doc"}

    # =========================
    # Content Limits
    # =========================
    MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "30000"))
    MAX_JD_CHARS = int(os.getenv("MAX_JD_CHARS", "12000"))
    MAX_RUBRIC_CHARS = int(os.getenv("MAX_RUBRIC_CHARS", "50000"))
    MAX_RESUME_CHARS_INFO_EXTRACTION = 15000

    # =========================
    # Scoring Settings
    # =========================
    PASS_THRESHOLD = int(os.getenv("PASS_THRESHOLD", "75"))
    MIN_SCORE_FOR_REPORT = PASS_THRESHOLD  # single source of truth

    # =========================
    # Field Mappings (CSV → Airtable)
    # =========================
    FIELD_MAP = {
        "organisation_id":   "organisation_id",
        "organisation_name": "organisation_name",
        "job_id":            "job_id",
        "job_name":          "job_name",
        "created_at":        "created_at",
        "updated_at":        "updated_at",
        "match_stage_name":  "match_stage_name",
        "candidate_id":      "candidate_id",
        "full_name":         "full_name",
        "email":             "email",
        "resume_file":       "resume_file",
        "cv_text":           "cv_text",
        "tier1_score":       "tier1_score",
        "ai_summary":        "ai_summary",
        "ai_strengths":      "ai_strengths",
        "ai_gaps":           "ai_gaps",
        "rubric_version":    "rubric_version",
        "rubric_hash":       "rubric_hash",
        "cache_key":         "cache_key",
    }

    # Field type definitions for upload normalisation
    TEXT_FIELDS = {
        "organisation_name",
        "job_name",
        "created_at", "updated_at",
        "match_stage_name",
        "full_name", "email",
        "resume_file",
        "cv_text",
        "ai_summary", "ai_strengths", "ai_gaps",
        "rubric_version", "rubric_hash",
        "cache_key",
    }

    NUMBER_FIELDS = {
        "ai_score", "tier1_score", "tier2_score",
        "organisation_id", "job_id", "candidate_id",
    }

    # =========================
    # Validation
    # =========================

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration is present."""
        required = {
            "OPENAI_API_KEY":  cls.OPENAI_API_KEY,
            "AIRTABLE_TOKEN":  cls.AIRTABLE_TOKEN,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}\n"
                f"Set these in your .env file or as environment variables."
            )

    @classmethod
    def validate_online_mode(cls) -> None:
        """Validate configuration for online mode (requires Manatal API)."""
        cls.validate()
        if not cls.MANATAL_API_TOKEN:
            raise ValueError(
                "MANATAL_API_TOKEN required for online mode.\n"
                "Set it in your .env file: MANATAL_API_TOKEN=your_token_here"
            )

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        cls.OFFLINE_INPUT_DIR.mkdir(exist_ok=True)
        cls.RUBRIC_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_rubric_path(cls, job_id: str) -> Path:
        json_path = cls.RUBRIC_DIR / f"rubric_{job_id}.json"
        if json_path.exists():
            return json_path
        return cls.RUBRIC_DIR / f"rubric_{job_id}.yaml"

    @classmethod
    def get_offline_json_path(cls, job_id: str) -> Path:
        return cls.OFFLINE_INPUT_DIR / f"job_{job_id}.json"

    @classmethod
    def get_scored_csv_path(cls, job_id: str, stage_name: str = None) -> Path:
        stage = stage_name or cls.TARGET_STAGE_NAME
        return cls.UPLOAD_DIR / f"manatal_job_{job_id}_{stage}_scored.csv"

    @classmethod
    def get_scored_json_path(cls, job_id: str, stage_name: str = None) -> Path:
        stage = stage_name or cls.TARGET_STAGE_NAME
        return cls.UPLOAD_DIR / f"manatal_job_{job_id}_{stage}_scored.json"
