# AI Recruitment Pipeline — Airtable + Render Frontend

A full-stack web service that wraps the AI recruitment scoring pipeline with a live dashboard.

## Features

- **Live streaming logs** — watch the pipeline run in real time via Server-Sent Events
- **Multi-job support** — add multiple job IDs and process them in one run
- **Candidate dashboard** — view Tier 1 & Tier 2 scores pulled directly from Airtable
- **Report viewer** — one-click access to detailed AI-generated candidate reports
- **Configurable** — adjust pass threshold, pipeline stage, and skip options per run

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Alpine.js + Tailwind CSS |
| Pipeline | Python subprocess (online_pipeline.py) |
| Data | Airtable REST API |

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/shaileshwaren/airtable-pipeline-render-frontend.git
cd airtable-pipeline-render-frontend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Run the server
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

## Deploying to Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect the GitHub repo
4. Set environment variables (see `.env.example`) in the Render dashboard
5. Deploy — Render auto-detects `render.yaml`

## Environment Variables

| Variable | Description |
|---|---|
| `MANATAL_API_TOKEN` | Manatal Open API token |
| `OPENAI_API_KEY` | OpenAI API key |
| `AIRTABLE_TOKEN` | Airtable Personal Access Token |
| `AIRTABLE_BASE_ID` | Airtable base ID (e.g. `appXXXX`) |
| `AIRTABLE_TABLE_ID` | Candidate table ID |
| `AIRTABLE_RUBRIC_TABLE_ID` | Rubric table ID |
| `AIRTABLE_JOB_TABLE_ID` | Job table ID |
| `PASS_THRESHOLD` | Min score for PASS status (default: 75) |
| `TARGET_STAGE_NAME` | Manatal stage to pull candidates from (default: New Candidates) |
| `TARGET_STAGE_AFTER` | Manatal stage to move candidates to after scoring (default: AI Screened) |
