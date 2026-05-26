"""
safety.py — Pre-LLM safety layer for prompt injection, PII detection, and data exfiltration.

This module uses ONLY regex and rule-based detection — no LLM calls.
LLMs can be jailbroken; regex cannot. This layer runs BEFORE the LLM sees
any ticket text, and can short-circuit the pipeline entirely for adversarial inputs.
"""

import re
from typing import Tuple, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# 1. PROMPT INJECTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

INJECTION_PATTERNS = [
    # ── Direct instruction override ──────────────────────────────────────
    r"ignore\s+(all\s+|previous\s+|prior\s+|above\s+|any\s+)?instructions",
    r"disregard\s+(your\s+|all\s+|any\s+|previous\s+)?instructions",
    r"override\s+(your\s+|all\s+|any\s+|the\s+)?",
    r"forget\s+(everything|all|your\s+instructions|the\s+rules)",
    r"new\s+(system\s+)?prompt",
    r"you\s+are\s+now\b",
    r"you\s+must\s+now\b",
    r"from\s+now\s+on\s+you",

    # ── Role-play / persona hijack ───────────────────────────────────────
    r"pretend\s+(you\s+are|to\s+be|you're)",
    r"act\s+as\s+(a\s+|an\s+)?(?!support|customer|agent)",
    r"roleplay\s+as",
    r"you\s+are\s+(?:DAN|a\s+hacker|an?\s+unrestricted|evil)",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"maintenance\s+mode",

    # ── System prompt extraction ─────────────────────────────────────────
    r"reveal\s+(your\s+|the\s+)?(system\s+)?prompt",
    r"what\s+are\s+your\s+instructions",
    r"show\s+(me\s+)?(your\s+|the\s+)?(system\s+)?prompt",
    r"(print|output|display|share)\s+(your\s+|the\s+)?(system\s+|internal\s+)?instructions",
    r"(print|output|display|share)\s+(your\s+|the\s+)?guidelines",
    r"what\s+(is|are)\s+your\s+(system\s+)?prompt",
    r"(tell|show)\s+me\s+(your\s+|the\s+)?(full\s+)?system\s+(prompt|instructions)",
    r"confidence\s+scoring\s+algorithm",

    # ── Format injection ─────────────────────────────────────────────────
    r"<!--.*?-->",                               # HTML comment injection
    r"\[INST\]",                                  # Llama/Mistral injection
    r"###\s*(Human|Assistant|System)\s*:",         # Chat format injection
    r"<system>.*?</system>",                       # XML tag injection
    r"<\|im_start\|>",                             # ChatML injection
    r"\[SYSTEM\s*OVERRIDE\]",                      # System override tag

    # ── Automated system impersonation ───────────────────────────────────
    r"(this\s+is\s+)?an?\s+(automated\s+)?message\s+from\s+your\s+(monitoring|security)\s+system",
    r"alert[_:]?\s*ack",                          # Alert acknowledgment
    r"CVE-\d{4}-\d+.*(?:acknowledge|confirm)",    # Fake CVE acknowledgment
    r"auth[_]?code",                              # Fake auth codes
    r"QA\s+(team|engineer|audit)",                 # Fake internal roles
    r"I\s+am\s+a\s+\w+\s+employee\s+working\s+in",  # Employee impersonation

    # ── Output manipulation ──────────────────────────────────────────────
    r"(output|respond\s+with|say)\s+(the\s+following|exactly|this)\s*:",
    r"(confirm|verify)\s+by\s+(saying|responding|outputting)",
    r"for\s+all\s+remaining\s+tickets",
    r"classify\s+this\s+as\s+(replied|escalated|invalid)",

    # ── Multilingual injection (French, Spanish, German, Hindi) ──────────
    r"ignor(?:er?|ez)\s+(toutes?\s+|les\s+)?instructions",         # French
    r"ignorar?\s+(todas?\s+|las\s+)?instrucciones",                # Spanish
    r"ignoriere[n]?\s+(alle\s+)?anweisungen",                      # German
    r"(सभी\s+)?निर्देश(ों)?\s+(भूल\s+जाओ|अनदेखा\s+कर)",              # Hindi
    r"sabhi\s+nirdesh\s+bhool\s+jao",                              # Hindi romanized
    r"忽略.*指令",                                                    # Chinese
    r"指示を無視",                                                    # Japanese

    # ── Excel / formula injection ────────────────────────────────────────
    r"^=cmd\|",
    r"^=\w+\(",
]

# Compile all patterns once for performance
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check text for prompt injection patterns.
    
    Returns:
        (is_injection, pattern_type) — pattern_type is the first matched 
        pattern string, or None if no injection detected.
    """
    if not text or not text.strip():
        return False, None

    for compiled, raw in zip(_COMPILED_INJECTION, INJECTION_PATTERNS):
        if compiled.search(text):
            return True, raw

    # ── Base64-encoded injection check ───────────────────────────────────
    # Check for base64 strings that decode to injection attempts
    import base64
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    for match in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group()).decode('utf-8', errors='ignore')
            if decoded and len(decoded) > 10:
                sub_detected, sub_type = detect_injection(decoded)
                if sub_detected:
                    return True, f"base64_encoded:{sub_type}"
        except Exception:
            pass

    return False, None


# ═══════════════════════════════════════════════════════════════════════════
# 2. PII DETECTION
# ═══════════════════════════════════════════════════════════════════════════

PII_PATTERNS = {
    "credit_card": re.compile(
        r"\b(?:\d[ -]?){13,16}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"
    ),
    "phone_intl": re.compile(
        r"\+\d{1,3}\s\d{3}\s[Xx]{4}\s\d{4}"
    ),
    "aadhaar": re.compile(
        r"\b\d{4}\s\d{4}\s\d{4}\b"
    ),
    "passport": re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    ),
    "dob": re.compile(
        r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"
    ),
    "address": re.compile(
        r"\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct)\b",
        re.IGNORECASE
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}

# Allowlisted "PII" that isn't real PII (common false positives)
_PII_ALLOWLIST = {
    "email": {
        "john@gmail.com",      # Example emails in generic text
        "john@company.com",
        "zhang.wei@example.com",
    }
}


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Detect PII in text.
    
    Returns:
        Dictionary mapping PII type to list of matched strings.
        Empty dict if no PII detected.
    """
    if not text:
        return {}

    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Filter out allowlisted items
            allowlist = _PII_ALLOWLIST.get(pii_type, set())
            filtered = [m for m in matches if m.strip().lower() not in {a.lower() for a in allowlist}]
            if filtered:
                found[pii_type] = filtered

    return found


def has_pii(text: str) -> bool:
    """Quick boolean check for PII presence."""
    return bool(detect_pii(text))


# ═══════════════════════════════════════════════════════════════════════════
# 3. PII MASKING (for LLM prompt — replace before sending)
# ═══════════════════════════════════════════════════════════════════════════

_MASK_REPLACEMENTS = {
    "credit_card": "[REDACTED_CARD]",
    "ssn": "[REDACTED_SSN]",
    "aadhaar": "[REDACTED_AADHAAR]",
    "passport": "[REDACTED_PASSPORT]",
    "dob": "[REDACTED_DOB]",
    "address": "[REDACTED_ADDRESS]",
    "ip_address": "[REDACTED_IP]",
    # We keep email and phone for context — the LLM may need them to
    # suggest password reset flows, etc. But we instruct the LLM not
    # to echo them in the response.
}


def mask_pii(text: str) -> str:
    """
    Replace sensitive PII with redaction tokens before sending to LLM.
    Keeps emails and phones visible for context (LLM is instructed not to echo).
    """
    if not text:
        return text

    masked = text
    for pii_type, replacement in _MASK_REPLACEMENTS.items():
        pattern = PII_PATTERNS.get(pii_type)
        if pattern:
            masked = pattern.sub(replacement, masked)
    return masked


# ═══════════════════════════════════════════════════════════════════════════
# 4. DATA EXFILTRATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

EXFIL_PATTERNS = [
    r"send\s+(me|to)\s+(all|the)\s+(data|tickets|corpus|documents)",
    r"list\s+all\s+(files|documents|users|tickets|articles)",
    r"dump\s+(the\s+)?(database|corpus|data)",
    r"export\s+(all|the)\s+(data|records|documents)",
    r"(complete\s+list|full\s+list|all)\s+of\s+support\s+articles",
    r"how\s+many\s+documents\s+you\s+have",
    r"(exact\s+)?retrieval\s+algorithm",
    r"(scrape|crawl|download)\s+(all\s+)?(support\s+)?(documentation|articles|docs)",
    r"save\s+it\s+as\s+a\s+local\s+dataset",
]

_COMPILED_EXFIL = [re.compile(p, re.IGNORECASE) for p in EXFIL_PATTERNS]


def detect_exfiltration(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect data exfiltration attempts.
    
    Returns:
        (is_exfil, pattern_type)
    """
    if not text:
        return False, None

    for compiled, raw in zip(_COMPILED_EXFIL, EXFIL_PATTERNS):
        if compiled.search(text):
            return True, raw

    return False, None


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED SAFETY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def run_safety_checks(text: str) -> dict:
    """
    Run all safety checks on text. Returns a dict with results.
    
    Returns:
        {
            "injection_detected": bool,
            "injection_type": str or None,
            "exfil_detected": bool,
            "exfil_type": str or None,
            "pii_detected": bool,
            "pii_types": dict,
            "masked_text": str,
            "should_block": bool,  # True = skip LLM, return escalated
        }
    """
    injection_detected, injection_type = detect_injection(text)
    exfil_detected, exfil_type = detect_exfiltration(text)
    pii_info = detect_pii(text)
    masked_text = mask_pii(text)

    return {
        "injection_detected": injection_detected,
        "injection_type": injection_type,
        "exfil_detected": exfil_detected,
        "exfil_type": exfil_type,
        "pii_detected": bool(pii_info),
        "pii_types": pii_info,
        "masked_text": masked_text,
        "should_block": injection_detected,  # Injections → block; exfil → let LLM handle with warning
    }
