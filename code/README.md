# MLE Support Triage Agent — Setup & Run Guide

## Prerequisites

- Python 3.10+ (tested with 3.12)
- API keys for Gemini and/or Groq

## Quick Start

```bash
# 1. Clone the repo and cd into it
cd MLE-hiring

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r code/requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   GOOGLE_API_KEY=your_gemini_key
#   GROQ_API_KEY=your_groq_key

# 6. Run the agent
python code/main.py

# 7. Validate output
python code/validate_output.py
```

## What It Does

The agent reads `support_tickets/support_tickets.csv` and produces `support_tickets/output.csv` with the following columns:

| Column | Description |
|---|---|
| issue | Original issue (preserved from input) |
| subject | Original subject (preserved from input) |
| company | Original company (preserved from input) |
| response | User-facing response grounded in corpus |
| product_area | Support category classification |
| status | `replied` or `escalated` |
| request_type | `product_issue`, `feature_request`, `bug`, or `invalid` |
| justification | Reasoning for the decision |
| confidence_score | 0.0-1.0 calibrated confidence |
| source_documents | Pipe-separated corpus file paths |
| risk_level | `low`, `medium`, `high`, or `critical` |
| pii_detected | `true` or `false` |
| language | ISO 639-1 language code |
| actions_taken | JSON array of tool calls |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Configuration

All configuration is in `code/config.py`:
- **Primary model:** `gemini-2.5-flash`
- **Fallback model:** `openai/gpt-oss-120b` (via Groq)
- **Temperature:** 0 (deterministic)
- **Retrieval:** BM25 with top-5 results

## Reproducibility

Running the agent twice on the same input produces identical output:
- `temperature=0` for all LLM calls
- `langdetect.DetectorFactory.seed = 42`
- Deterministic BM25 retrieval
- No randomness anywhere in the pipeline

## Troubleshooting

- **Gemini rate limits:** The agent retries with exponential backoff (2s, 4s, 8s). If all retries fail, it falls back to Groq.
- **Missing .env:** Copy `.env.example` to `.env` and add your keys.
- **Import errors:** Make sure you installed requirements: `pip install -r code/requirements.txt`
