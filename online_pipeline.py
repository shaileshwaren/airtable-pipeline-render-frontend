#!/usr/bin/env python3
"""online_pipeline.py

ONLINE PIPELINE - Multi-Job Mode
Fetches candidates from Manatal API and processes multiple jobs

Workflow (per job):
0) Rubric (generate_rubric.py) - checks Airtable cache; generates + stores if not found
1) AI scoring (python8.py)     - fetches from Manatal API, scores against rubric
2) Upload to Airtable (upload_airtable.py)
3) Generate detailed reports (generate_detailed_reports.py)

Usage:
  # Single job
  python online_pipeline.py 3419430

  # Multiple jobs (comma-separated)
  python online_pipeline.py "3419430, 3261113"

  # With optional flags
  python online_pipeline.py 3419430 --skip-upload --skip-reports
  python online_pipeline.py 3419430 --skip-rubric   # skip rubric step entirely
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON8 = HERE / "python8.py"
UPLOAD_AIRTABLE = HERE / "upload_airtable.py"
GENERATE_REPORTS = HERE / "generate_detailed_reports.py"
GENERATE_RUBRIC = HERE / "generate_rubric.py"
CONFIG_FILE = HERE / "online_config.txt"
ADVANCED_CONFIG_FILE = HERE / "online_advanced_config.txt"


def run_step(step_num: int, total_steps: int, description: str, cmd: list[str]) -> int:
    """Run a pipeline step. Returns the process exit code."""
    print(f"\n{'='*70}")
    print(f"[STEP {step_num}/{total_steps}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode not in (0, 2):
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def warn_missing_env() -> None:
    required = ["OPENAI_API_KEY", "AIRTABLE_TOKEN", "MANATAL_API_TOKEN"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(
            "\n[WARN] Missing environment variables: "
            + ", ".join(missing)
            + "\n       Set them in .env file or as environment variables.\n"
        )


def validate_files_exist() -> None:
    scripts = {
        "python8.py": PYTHON8,
        "upload_airtable.py": UPLOAD_AIRTABLE,
        "generate_detailed_reports.py": GENERATE_REPORTS,
    }
    missing = [f"{n} not found at {p}" for n, p in scripts.items() if not p.exists()]
    if missing:
        print("ERROR: Missing required scripts:")
        for msg in missing:
            print(f"  - {msg}")
        print(f"\nEnsure all scripts are in the same folder: {HERE}")
        sys.exit(2)


def load_config() -> dict:
    config = {}

    if not CONFIG_FILE.exists():
        return {
            "stage_name": "New Candidates",
            "skip_scoring": False,
            "skip_upload": False,
            "generate_reports": True,
        }

    for cfg_file in [CONFIG_FILE, ADVANCED_CONFIG_FILE]:
        if not cfg_file.exists():
            continue
        with cfg_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    print(f"[WARN] Line {line_num} ignored (no '=' found): {line}")
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not value:
                    continue
                if value.lower() in ("true", "yes", "1"):
                    value = True
                elif value.lower() in ("false", "no", "0"):
                    value = False
                elif value.isdigit():
                    value = int(value)
                config[key] = value

    return config


def process_single_job(job_id: str, config: dict, global_args: argparse.Namespace) -> bool:
    """Process a single job. Returns True on success."""

    stage_name = config.get("stage_name", "New Candidates")
    skip_rubric = global_args.skip_rubric or config.get("skip_rubric", False)
    skip_scoring = global_args.skip_scoring or config.get("skip_scoring", False)
    skip_upload = global_args.skip_upload or config.get("skip_upload", False)
    skip_reports = global_args.skip_reports or config.get("generate_reports", True) == False

    print(f"\n{'='*70}")
    print(f"Processing Job: {job_id}")
    print(f"{'='*70}")
    print(f"Stage: {stage_name}")
    print(f"INFO: Job name, org ID, and org name will be fetched from Manatal API")
    print(f"{'='*70}\n")

    total_steps = sum([not skip_rubric, not skip_scoring, not skip_upload, not skip_reports])
    step_num = 1

    try:
        # ── STEP 0: Rubric ───────────────────────────────────────────────
        rubric_regenerated = False
        if not skip_rubric:
            rubric_exit = run_step(
                step_num, total_steps,
                f"Rubric check/generate for job {job_id}",
                [sys.executable, str(GENERATE_RUBRIC), str(job_id)],
            )
            # Exit code 2 = rubric was (re)generated → force rescore of all candidates
            rubric_regenerated = (rubric_exit == 2)
            if rubric_regenerated:
                print(f"\n[INFO] Rubric was regenerated for job {job_id} — all candidates will be rescored.")
            step_num += 1
        else:
            print("\nSkipped: Rubric step")

        # ── STEP 1: AI Scoring ───────────────────────────────────────────
        if not skip_scoring:
            scoring_cmd = [sys.executable, str(PYTHON8), str(job_id)]
            if rubric_regenerated:
                scoring_cmd.append("--force-rescore")
            run_step(
                step_num, total_steps,
                "AI Scoring (fetch from Manatal + score against rubric)"
                + (" [FORCE RESCORE]" if rubric_regenerated else ""),
                scoring_cmd,
            )
            step_num += 1
        else:
            print("\nSkipped: AI Scoring")

        # ── STEP 2: Upload to Airtable ───────────────────────────────────
        if not skip_upload:
            run_step(
                step_num, total_steps,
                "Upload to Airtable",
                [sys.executable, str(UPLOAD_AIRTABLE), str(job_id)],
            )
            step_num += 1
        else:
            print("\nSkipped: Airtable Upload")

        # ── STEP 3: Generate Detailed Reports ────────────────────────────
        if not skip_reports:
            run_step(
                step_num, total_steps,
                "Generate Detailed Reports",
                [sys.executable, str(GENERATE_REPORTS), str(job_id)],
            )
        else:
            print("\nSkipped: Detailed Reports")

        print(f"\n{'='*70}")
        print(f"Job {job_id} completed successfully!")
        print(f"{'='*70}\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Job {job_id} failed")
        print(f"{'='*70}")
        print(f"Exit code: {e.returncode}")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"\nContinuing with next job...\n")
        return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Online Pipeline: Fetch and process candidates from Manatal API → Airtable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single job
  python online_pipeline.py 3419430

  # Process multiple jobs (comma-separated)
  python online_pipeline.py "3419430, 3261113, 3600123"

  # Skip specific steps
  python online_pipeline.py 3419430 --skip-upload --skip-reports
  python online_pipeline.py 3419430 --skip-rubric
        """,
    )

    parser.add_argument("job_ids", help="Job IDs to process (comma-separated for multiple)")
    parser.add_argument("--skip-rubric", action="store_true", help="Skip auto-rubric generation (use existing rubric or fail)")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip AI scoring step")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Airtable upload step")
    parser.add_argument("--skip-reports", action="store_true", help="Skip detailed report generation")

    args = parser.parse_args(argv[1:])

    validate_files_exist()

    job_ids = [jid.strip() for jid in args.job_ids.split(",")]

    print(f"\n{'='*70}")
    print("ONLINE PIPELINE - AIRTABLE MODE")
    print(f"{'='*70}\n")
    print(f"Processing {len(job_ids)} job(s): {', '.join(job_ids)}\n")

    warn_missing_env()
    config = load_config()

    successful_jobs = []
    failed_jobs = []

    for idx, job_id in enumerate(job_ids, 1):
        print(f"\n{'='*70}")
        print(f"Processing Job {idx} of {len(job_ids)}: {job_id}")
        print(f"{'='*70}\n")

        try:
            success = process_single_job(job_id, config, args)
            if success:
                successful_jobs.append(job_id)
            else:
                failed_jobs.append(job_id)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"       Skipping job {job_id}\n")
            failed_jobs.append(job_id)
        except Exception as e:
            print(f"ERROR: Unexpected error for job {job_id}: {e}")
            print(f"       Skipping job {job_id}\n")
            failed_jobs.append(job_id)

    print(f"\n{'='*70}")
    print("MULTI-JOB PIPELINE SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total Jobs: {len(job_ids)}")
    print(f"Successful: {len(successful_jobs)}")
    print(f"Failed: {len(failed_jobs)}\n")

    if successful_jobs:
        print("Successful Jobs:")
        for job_id in successful_jobs:
            print(f"  - {job_id}")
        print()

    if failed_jobs:
        print("Failed Jobs:")
        for job_id in failed_jobs:
            print(f"  - {job_id}")
        print()

    print(f"{'='*70}\n")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except KeyboardInterrupt:
        print("\n\nWARNING: Pipeline interrupted by user")
        raise SystemExit(130)
