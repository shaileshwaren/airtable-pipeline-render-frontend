#!/usr/bin/env python3
"""main.py — FastAPI web service for the AI Recruitment Pipeline.

Exposes:
  POST /api/run              — start a pipeline run, returns run_id
  GET  /api/stream/{run_id}  — SSE stream of live pipeline logs
  GET  /api/status/{run_id}  — current run status
  GET  /api/candidates/{job_id} — fetch candidates from Airtable
  GET  /api/jobs             — list all jobs from Airtable
  GET  /                     — serve the frontend
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

app = FastAPI(title="AI Recruitment Pipeline", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory run store  {run_id → run_state}
# ─────────────────────────────────────────────────────────────────────────────
runs: Dict[str, Dict[str, Any]] = {}


class RunRequest(BaseModel):
    job_ids: List[str]
    pass_threshold: int = 75
    stage_name: str = "New Candidates"
    skip_rubric: bool = False
    skip_reports: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Airtable helpers
# ─────────────────────────────────────────────────────────────────────────────
def _at_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {os.getenv('AIRTABLE_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def _at_get_records(table_id: str, formula: Optional[str] = None) -> List[Dict]:
    base_id = os.getenv("AIRTABLE_BASE_ID", "")
    token = os.getenv("AIRTABLE_TOKEN", "")
    if not base_id or not token:
        return []

    records: List[Dict] = []
    offset: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"pageSize": 100}
        if formula:
            params["filterByFormula"] = formula
        if offset:
            params["offset"] = offset

        try:
            r = http_requests.get(
                f"https://api.airtable.com/v0/{base_id}/{table_id}",
                headers=_at_headers(),
                params=params,
                timeout=30,
            )
            if not r.ok:
                break
            data = r.json()
            records.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break
        except Exception:
            break

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (background thread)
# ─────────────────────────────────────────────────────────────────────────────
def _pipeline_thread(run_id: str, req: RunRequest) -> None:
    q: queue.Queue = runs[run_id]["queue"]

    env = {**os.environ}
    env["TARGET_STAGE_NAME"] = req.stage_name
    env["PASS_THRESHOLD"] = str(req.pass_threshold)
    env["MIN_SCORE_FOR_REPORT"] = str(req.pass_threshold)
    # Force line-by-line (unbuffered) output from all Python subprocesses
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    job_ids_str = ", ".join(req.job_ids)
    # -u flag: run Python in unbuffered mode so each print() flushes immediately
    cmd = [sys.executable, "-u", str(HERE / "online_pipeline.py"), job_ids_str]
    if req.skip_rubric:
        cmd.append("--skip-rubric")
    if req.skip_reports:
        cmd.append("--skip-reports")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,          # line-buffered: flush after every newline
            env=env,
            cwd=str(HERE),
        )
        runs[run_id]["process"] = proc

        # readline() reads one line at a time as soon as it arrives,
        # unlike iterating proc.stdout which can internally buffer reads
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            q.put(line.rstrip())

        proc.wait()
        runs[run_id]["exit_code"] = proc.returncode
        runs[run_id]["status"] = "success" if proc.returncode == 0 else "failed"

    except Exception as exc:
        q.put(f"[ERROR] {exc}")
        runs[run_id]["status"] = "failed"
    finally:
        runs[run_id]["done"] = True
        q.put(None)  # sentinel — tells the SSE generator to close


# ─────────────────────────────────────────────────────────────────────────────
# API — pipeline control
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/run")
def start_run(req: RunRequest):
    if not req.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")

    run_id = str(uuid.uuid4())[:8]
    runs[run_id] = {
        "queue": queue.Queue(),
        "process": None,
        "done": False,
        "status": "running",
        "exit_code": None,
        "job_ids": req.job_ids,
    }

    threading.Thread(
        target=_pipeline_thread,
        args=(run_id, req),
        daemon=True,
    ).start()

    return {"run_id": run_id, "status": "started"}


@app.get("/api/stream/{run_id}")
def stream_logs(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    def generate():
        q: queue.Queue = runs[run_id]["queue"]
        while True:
            try:
                line = q.get(timeout=25)
            except queue.Empty:
                # Heartbeat to keep the connection alive
                yield "data: [heartbeat]\n\n"
                continue

            if line is None:
                status = runs[run_id].get("status", "done")
                yield f"data: [PIPELINE_DONE:{status}]\n\n"
                break

            # Each SSE message must be a single line
            safe = line.replace("\r", "").replace("\n", " ")
            yield f"data: {safe}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/status/{run_id}")
def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = runs[run_id]
    return {
        "run_id": run_id,
        "status": run["status"],
        "done": run["done"],
        "exit_code": run["exit_code"],
        "job_ids": run["job_ids"],
    }


@app.post("/api/cancel/{run_id}")
def cancel_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    proc = runs[run_id].get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        runs[run_id]["status"] = "cancelled"
    return {"status": "cancelled"}


# ─────────────────────────────────────────────────────────────────────────────
# API — Airtable data
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/candidates/{job_id}")
def get_candidates(job_id: str):
    table_id = os.getenv("AIRTABLE_TABLE_ID", "")
    if not table_id:
        raise HTTPException(status_code=500, detail="AIRTABLE_TABLE_ID not configured")

    records = _at_get_records(table_id, f"{{job_id}}={job_id}")

    candidates = []
    for rec in records:
        f = rec.get("fields", {})

        cv_att = f.get("CV") or f.get("cv") or []
        cv_url = cv_att[0].get("url") if cv_att else None

        rpt_att = f.get("ai_report_html") or []
        report_url = rpt_att[0].get("url") if rpt_att else None

        candidates.append({
            "candidate_id":      f.get("candidate_id"),
            "full_name":         f.get("full_name", "—"),
            "email":             f.get("email", ""),
            "job_name":          f.get("job_name", ""),
            "org_name":          f.get("organisation_name", ""),
            "match_stage_name":  f.get("match_stage_name", ""),
            "tier1_score":       f.get("tier1_score"),
            "tier2_score":       f.get("tier2_score"),
            "ai_summary":        f.get("ai_summary", ""),
            "ai_strengths":      f.get("ai_strengths", ""),
            "ai_gaps":           f.get("ai_gaps", ""),
            "cv_url":            cv_url,
            "report_url":        report_url,
        })

    candidates.sort(key=lambda c: c.get("tier1_score") or 0, reverse=True)
    return {"candidates": candidates, "total": len(candidates)}


@app.get("/api/jobs")
def get_jobs():
    job_table_id = os.getenv("AIRTABLE_JOB_TABLE_ID", "tblCV6w4fGex9VgzK")
    records = _at_get_records(job_table_id)

    jobs = []
    for rec in records:
        f = rec.get("fields", {})
        jobs.append({
            "job_id":      f.get("job_id"),
            "job_name":    f.get("job_name", ""),
            "client_name": f.get("client_name", ""),
            "word_cnt":    f.get("word_cnt"),
        })

    jobs.sort(key=lambda j: j.get("job_id") or 0, reverse=True)
    return {"jobs": jobs}


# ─────────────────────────────────────────────────────────────────────────────
# Static files + catch-all → index.html
# ─────────────────────────────────────────────────────────────────────────────
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"error": "index.html not found in static/"}
    return FileResponse(str(index))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
