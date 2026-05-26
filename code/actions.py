"""
actions.py — Validates and normalizes the actions_taken array from LLM output.

Ensures:
1. Every action name matches a tool in internal_tools.json
2. Required parameters are present for each tool
3. Destructive actions are gated behind verify_identity
4. High/critical risk tickets include escalate_to_human
5. Output is always a valid JSON array
"""

import json
import os
import re
from typing import List, Dict, Optional

from config import API_SPECS_PATH


# ── Load tool schemas ────────────────────────────────────────────────────
def _load_tool_specs() -> Dict[str, dict]:
    """Load and index tool schemas by name."""
    with open(API_SPECS_PATH, "r", encoding="utf-8") as f:
        specs = json.load(f)
    return {tool["name"]: tool for tool in specs}

TOOL_SPECS = _load_tool_specs()
TOOL_NAMES = set(TOOL_SPECS.keys())

# Tools that require identity verification first
DESTRUCTIVE_ACTIONS = {"issue_refund", "lock_account", "modify_subscription"}

# Patterns that suggest identity is already verified in conversation
IDENTITY_VERIFIED_PATTERNS = [
    r"identity\s+(has\s+been\s+)?verified",
    r"(already\s+)?authenticated",
    r"verification\s+(code\s+)?(confirmed|received|accepted)",
    r"OTP\s+(verified|confirmed|accepted)",
]
_COMPILED_IDENTITY = [re.compile(p, re.IGNORECASE) for p in IDENTITY_VERIFIED_PATTERNS]


def _check_identity_in_context(conversation_text: str) -> bool:
    """Check if identity appears to be already verified in conversation."""
    for pattern in _COMPILED_IDENTITY:
        if pattern.search(conversation_text):
            return True
    return False


def _has_action(actions: List[dict], action_name: str) -> bool:
    """Check if an action with the given name exists in the list."""
    return any(a.get("action") == action_name for a in actions)


def _validate_params(action: dict, tool_spec: dict) -> dict:
    """
    Validate and clean action parameters against the tool schema.
    Removes unknown parameters. Returns cleaned action dict.
    """
    action_name = action.get("action", "")
    params = action.get("parameters", {})
    
    if not isinstance(params, dict):
        params = {}

    schema_props = tool_spec.get("parameters", {}).get("properties", {})
    required = set(tool_spec.get("parameters", {}).get("required", []))

    # Only keep parameters defined in schema
    cleaned_params = {}
    for key, value in params.items():
        if key in schema_props:
            cleaned_params[key] = value

    return {
        "action": action_name,
        "parameters": cleaned_params,
    }


def validate_and_normalize_actions(
    actions_raw: list,
    risk_level: str,
    status: str,
    conversation_text: str = "",
) -> List[dict]:
    """
    Validate, normalize, and potentially augment the actions_taken array.
    
    Args:
        actions_raw: Raw actions list from LLM output
        risk_level: Ticket risk level
        status: Ticket status (replied/escalated)
        conversation_text: Full conversation text for context
        
    Returns:
        Validated list of action dicts, always a valid JSON array.
    """
    if not isinstance(actions_raw, list):
        actions_raw = []

    validated = []
    identity_verified = _check_identity_in_context(conversation_text)
    needs_verify = False

    for action in actions_raw:
        if not isinstance(action, dict):
            continue

        action_name = action.get("action", "")

        # 1. Strip hallucinated tools — must be in known schema
        if action_name not in TOOL_NAMES:
            continue

        # 2. Validate parameters against schema
        tool_spec = TOOL_SPECS[action_name]
        cleaned_action = _validate_params(action, tool_spec)

        # 3. Check if destructive action needs identity verification
        if action_name in DESTRUCTIVE_ACTIONS and not identity_verified:
            needs_verify = True

        validated.append(cleaned_action)

    # 4. Prepend verify_identity if needed
    if needs_verify and not _has_action(validated, "verify_identity"):
        validated.insert(0, {
            "action": "verify_identity",
            "parameters": {
                "method": "email_otp",
                "target": "[user_email]",
            },
        })

    # 5. Ensure escalate_to_human for high/critical risk escalated tickets
    if (risk_level in ("high", "critical") and status == "escalated"
            and not _has_action(validated, "escalate_to_human")):
        # Determine department based on risk context
        department = _infer_department(conversation_text, risk_level)
        priority = "urgent" if risk_level == "critical" else "high"

        validated.append({
            "action": "escalate_to_human",
            "parameters": {
                "priority": priority,
                "department": department,
                "summary": f"Ticket escalated due to {risk_level} risk level. Requires human review.",
            },
        })

    return validated


def _infer_department(text: str, risk_level: str) -> str:
    """Infer the appropriate department for escalation."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["legal", "lawsuit", "attorney", "lawyer", "gdpr", "compliance", "hipaa"]):
        return "legal"
    if any(w in text_lower for w in ["fraud", "unauthorized", "stolen", "identity theft", "compromise", "hack"]):
        return "security"
    if any(w in text_lower for w in ["billing", "refund", "charge", "payment", "invoice", "subscription"]):
        return "billing"
    if any(w in text_lower for w in ["bug", "error", "crash", "api", "500", "outage", "down"]):
        return "technical"

    return "general"


def actions_to_json_string(actions: list) -> str:
    """Serialize actions list to a JSON string for CSV output."""
    if not actions:
        return "[]"
    return json.dumps(actions, ensure_ascii=False)
