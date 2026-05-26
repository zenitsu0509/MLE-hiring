"""
main.py — Main orchestrator for the MLE Support Triage Agent.

Pipeline per ticket:
  1. Parse issue JSON (multi-turn conversation)
  2. Safety check (injection, exfiltration, PII)
  3. Language detection
  4. BM25 retrieval (domain-filtered)
  5. LLM call (Gemini 2.5 Flash → Groq fallback)
  6. Post-processing (validate sources, actions, confidence)
  7. Write output CSV

Never crashes — every ticket produces a valid row, guaranteed.
"""

import json
import sys
import os
import time
import traceback
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Ensure code/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUPPORT_TICKETS_PATH, OUTPUT_CSV_PATH,
    COMPANY_TO_DOMAIN, REPO_ROOT,
)
from safety import run_safety_checks
from retriever import CorpusRetriever
from agent import call_llm
from actions import validate_and_normalize_actions, actions_to_json_string


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect the primary language of text. Returns ISO 639-1 code.
    Falls back to 'en' on error.
    """
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42  # Deterministic
        lang = detect(text)
        # Normalize: langdetect may return 'zh-cn' etc.
        return lang.split("-")[0].lower()[:2]
    except Exception:
        return "en"


# ═══════════════════════════════════════════════════════════════════════════
# TICKET PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_issue(issue_raw: str) -> str:
    """
    Parse the issue column (JSON array of conversation turns).
    Returns concatenated text of all turns.
    """
    if not issue_raw or not str(issue_raw).strip():
        return ""

    issue_str = str(issue_raw).strip()

    try:
        conversation = json.loads(issue_str)

        if isinstance(conversation, list):
            parts = []
            for msg in conversation:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"[{role}]: {content}")
                elif isinstance(msg, str):
                    parts.append(msg)
            return "\n".join(parts)

        elif isinstance(conversation, str):
            return conversation

    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — treat as plain text
        return issue_str

    return issue_str


def extract_user_messages(issue_raw: str) -> str:
    """Extract only user messages for retrieval query (skip agent responses)."""
    if not issue_raw or not str(issue_raw).strip():
        return ""

    try:
        conversation = json.loads(str(issue_raw).strip())
        if isinstance(conversation, list):
            user_parts = []
            for msg in conversation:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_parts.append(msg.get("content", ""))
            return " ".join(user_parts)
    except Exception:
        pass

    return str(issue_raw)


# ═══════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def make_injection_response(row: dict, injection_type: str,
                            pii_detected: bool, language: str) -> dict:
    """
    Build a response for detected prompt injection.
    Skips LLM entirely — pure rule-based.
    """
    return {
        "issue": row.get("Issue", ""),
        "subject": row.get("Subject", ""),
        "company": row.get("Company", ""),
        "response": (
            "I've identified that this message contains content that appears to be "
            "an attempt to manipulate the support system. This request has been "
            "flagged and escalated to our security team for review. If you have a "
            "legitimate support question, please submit a new ticket with your "
            "actual inquiry."
        ),
        "product_area": "security",
        "status": "escalated",
        "request_type": "invalid",
        "justification": (
            f"Prompt injection/adversarial input detected by safety layer. "
            f"Pattern: {injection_type}. Ticket escalated without LLM processing "
            f"to prevent any compliance with malicious instructions."
        ),
        "confidence_score": 0.95,
        "source_documents": "",
        "risk_level": "critical",
        "pii_detected": str(pii_detected).lower(),
        "language": language,
        "actions_taken": json.dumps([{
            "action": "escalate_to_human",
            "parameters": {
                "priority": "urgent",
                "department": "security",
                "summary": "Prompt injection attempt detected. Escalated for security review."
            }
        }]),
    }


def make_fallback_response(row: dict, error_msg: str) -> dict:
    """
    Build a safe fallback response when the pipeline fails.
    Ensures we NEVER crash — every ticket gets a row.
    """
    return {
        "issue": row.get("Issue", ""),
        "subject": row.get("Subject", ""),
        "company": row.get("Company", ""),
        "response": (
            "We apologize for the inconvenience. Your request has been escalated "
            "to our support team for manual review. A representative will follow "
            "up with you shortly."
        ),
        "product_area": "general_support",
        "status": "escalated",
        "request_type": "product_issue",
        "justification": (
            f"Automated processing encountered an error. Escalated for human review. "
            f"Error: {error_msg[:200]}"
        ),
        "confidence_score": 0.15,
        "source_documents": "",
        "risk_level": "medium",
        "pii_detected": "false",
        "language": "en",
        "actions_taken": json.dumps([{
            "action": "escalate_to_human",
            "parameters": {
                "priority": "normal",
                "department": "general",
                "summary": "Automated processing failed. Requires manual review."
            }
        }]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_ticket(row: dict, retriever: CorpusRetriever,
                   valid_paths: set, ticket_num: int) -> dict:
    """Process a single support ticket through the full pipeline."""

    # ── Step 1: Parse issue ──────────────────────────────────────────────
    issue_raw = row.get("Issue", "")
    full_text = parse_issue(issue_raw)
    subject = str(row.get("Subject", "") or "")
    company = str(row.get("Company", "") or "").strip()

    # Combine subject + issue for full context
    combined_text = f"{subject}\n{full_text}".strip()

    if not combined_text:
        # Empty ticket
        return {
            "issue": issue_raw,
            "subject": subject,
            "company": company,
            "response": "It seems your message was empty. Please submit a new ticket with your support question and we'll be happy to help.",
            "product_area": "general_support",
            "status": "replied",
            "request_type": "invalid",
            "justification": "Empty ticket with no content. Replied with a request to resubmit.",
            "confidence_score": 0.95,
            "source_documents": "",
            "risk_level": "low",
            "pii_detected": "false",
            "language": "en",
            "actions_taken": "[]",
        }

    # ── Step 2: Safety checks ────────────────────────────────────────────
    safety = run_safety_checks(combined_text)
    language = detect_language(full_text)

    if safety["should_block"]:
        print(f"  ⚠️  Ticket {ticket_num}: INJECTION DETECTED → escalating")
        return make_injection_response(
            row, safety["injection_type"],
            safety["pii_detected"], language
        )

    # ── Step 3: Retrieval ────────────────────────────────────────────────
    # Use user messages only for retrieval query (cleaner signal)
    query_text = extract_user_messages(issue_raw) or combined_text
    # Also include subject for retrieval
    query_text = f"{subject} {query_text}".strip()

    # Determine domain for pre-filtering
    domain = None
    if company:
        domain = COMPANY_TO_DOMAIN.get(company.lower())

    retrieved = retriever.search(query_text, domain=domain)
    corpus_context = retriever.format_context(retrieved)
    retrieved_paths = [r[0]["filepath"] for r in retrieved if r[1] > 0.05]
    # CRITICAL: Only valid paths
    retrieved_paths = [p for p in retrieved_paths if p in valid_paths]

    max_bm25_score = max((r[1] for r in retrieved), default=0.0)

    # ── Step 4: LLM call ─────────────────────────────────────────────────
    result = call_llm(
        ticket_text=safety["masked_text"],
        corpus_context=corpus_context,
        valid_paths=retrieved_paths,
        pii_detected=safety["pii_detected"],
        language=language,
        exfil_warning=safety["exfil_detected"],
        max_bm25_score=max_bm25_score,
    )

    # ── Step 5: Post-process ─────────────────────────────────────────────

    # Validate source_documents — only keep paths that actually exist
    sources_raw = result.get("source_documents", "")
    if sources_raw:
        valid_sources = [s.strip() for s in str(sources_raw).split("|")
                         if s.strip() and s.strip() in valid_paths]
        result["source_documents"] = "|".join(valid_sources)
    else:
        result["source_documents"] = ""

    # Validate and normalize actions
    actions_raw = result.get("actions_taken", [])
    actions_validated = validate_and_normalize_actions(
        actions_raw,
        risk_level=result.get("risk_level", "low"),
        status=result.get("status", "replied"),
        conversation_text=combined_text,
    )

    # ── Step 6: Format final output row ──────────────────────────────────
    return {
        "issue": issue_raw,
        "subject": subject,
        "company": company,
        "response": result.get("response", ""),
        "product_area": result.get("product_area", "general_support"),
        "status": result.get("status", "escalated"),
        "request_type": result.get("request_type", "product_issue"),
        "justification": result.get("justification", ""),
        "confidence_score": result.get("confidence_score", 0.5),
        "source_documents": result.get("source_documents", ""),
        "risk_level": result.get("risk_level", "medium"),
        "pii_detected": str(result.get("pii_detected", False)).lower(),
        "language": result.get("language", language),
        "actions_taken": actions_to_json_string(actions_validated),
    }


def main():
    """Main entry point — process all tickets and write output.csv."""
    start_time = time.time()

    print("=" * 60)
    print("MLE Support Triage Agent")
    print("=" * 60)

    # ── Initialize retriever ─────────────────────────────────────────────
    print("\n[1/3] Building corpus index...")
    retriever = CorpusRetriever()
    valid_paths = retriever.get_valid_paths()
    print(f"       Valid paths: {len(valid_paths)}")

    # ── Load tickets ─────────────────────────────────────────────────────
    print(f"\n[2/3] Loading tickets from {SUPPORT_TICKETS_PATH}...")
    df = pd.read_csv(SUPPORT_TICKETS_PATH)
    print(f"       Loaded {len(df)} tickets")

    import concurrent.futures

    # ── Process tickets ──────────────────────────────────────────────────
    print(f"\n[3/3] Processing tickets with ThreadPoolExecutor...\n")
    results = [None] * len(df)

    def process_wrapper(args):
        idx, row_dict = args
        ticket_num = idx + 1
        try:
            res = process_ticket(row_dict, retriever, valid_paths, ticket_num)
            return idx, res
        except Exception as e:
            print(f"  ❌ Ticket {ticket_num} FAILED: {e}")
            traceback.print_exc()
            return idx, make_fallback_response(row_dict, str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(process_wrapper, (idx, row.to_dict())))
            
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Processing"):
            idx, res = future.result()
            results[idx] = res

    # ── Write output ─────────────────────────────────────────────────────
    output_df = pd.DataFrame(results)

    # Ensure column order matches expected output
    expected_columns = [
        "issue", "subject", "company", "response", "product_area",
        "status", "request_type", "justification", "confidence_score",
        "source_documents", "risk_level", "pii_detected", "language",
        "actions_taken",
    ]
    output_df = output_df[expected_columns]
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    replied = sum(1 for r in results if r.get("status") == "replied")
    escalated = sum(1 for r in results if r.get("status") == "escalated")

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(results)} tickets processed in {elapsed:.1f}s")
    print(f"  Replied:   {replied}")
    print(f"  Escalated: {escalated}")
    print(f"  Output:    {OUTPUT_CSV_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
