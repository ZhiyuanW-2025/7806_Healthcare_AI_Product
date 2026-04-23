import base64
import html
import io
import json
import math
import os
import re
import time
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_PATH = BASE_DIR / "data" / "mock_resources.json"
VILLAGES_PATH = BASE_DIR / "data" / "villages.json"
KB_PATH = BASE_DIR / "data" / "eu_health_kb.json"
MEDICAL_KB_PATH = BASE_DIR / "data" / "medical_kb.json"
MEDICAL_INDEX_PATH = BASE_DIR / "data" / "medical_kb_index.json"
LOW_CONFIDENCE_THRESHOLD = 0.85

SYSTEM_PROMPT = """
You are RuralLink Agent, the chief rural healthcare coordinator for EU community settings.
Tone: professional, calm, and empathetic.
Rules:
- Role boundary: you are a medical education and triage assistant, not a diagnosing or prescribing doctor.
- You may provide symptom explanation, OTC general knowledge, and care-seeking guidance.
- Do not provide prescription decisions, final doctor-equivalent diagnosis, or patient-specific dosing when information is insufficient.
- Never claim a real dispatch happened unless user explicitly confirms and the system executes mock dispatch.
- Information first, location second: for general medicine/health questions, answer guidance first. Do not ask address or show map by default.
- When asked about medicine, provide general OTC education first, add a safety warning to confirm with a pharmacist, then optionally ask whether user wants nearby pharmacy lookup.
- Strictly avoid Geo_RAG/map unless user explicitly asks location/distance, mentions a transport barrier, or triage is high/urgent.
- Use a short intake frame where relevant: symptoms, duration, severity, past history, allergies, current medications, age/pregnancy.
- Provide risk stratification before recommendations.
- If red-flag emergency signs are present (e.g., chest pain, breathing difficulty, stroke signs, confusion), escalate to emergency immediately and avoid long dialogue.
- Output language: English.
""".strip()


# ---------- UI helpers ----------
def apply_ui_theme() -> None:
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] {
  background: #e4eee6;
}
[data-testid="stAppViewContainer"] .main .block-container {
  padding-bottom: 120px;
}
.card {
  background: #ffffff;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(31, 68, 51, 0.12);
  padding: 1rem 1.1rem;
  border: 1px solid #c9dccc;
  margin-bottom: 1rem;
}
.agent-zone {
  background: linear-gradient(130deg, #e4eee6 0%, #f4f8f4 100%);
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(31, 68, 51, 0.12);
  padding: 1rem 1.1rem;
  border: 1px solid #c9dccc;
  margin-bottom: 1rem;
}
.stButton > button {
  background: #1F4433;
  color: #fff;
  border-radius: 10px;
  border: 1px solid #1F4433;
  font-weight: 600;
}
.stButton > button:hover {
  background: #b05b24;
  border-color: #b05b24;
}
.pulse {
  display: inline-block;
  padding: 0.45rem 0.8rem;
  border-radius: 999px;
  border: 1px solid #b8cfbe;
  color: #1F4433;
  background: #e4eee6;
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0% { transform: scale(1); opacity: 0.75; }
  50% { transform: scale(1.04); opacity: 1; }
  100% { transform: scale(1); opacity: 0.75; }
}

/* Input console: keep 3 input boxes same visual height */
.st-key-input_console {
  --input-box-h: 96px;
}
.st-key-input_console [data-testid="stFileUploaderDropzone"] {
  height: var(--input-box-h) !important;
  min-height: var(--input-box-h) !important;
  padding-top: 8px !important;
  padding-bottom: 8px !important;
  position: relative !important;
}
.st-key-input_console [data-testid="stFileUploaderDropzoneInstructions"] {
  display: none !important;
}
.st-key-input_console [data-testid="stFileUploaderDropzone"]::after {
  content: "Supported file types: JPG, JPEG, PNG, PDF";
  position: absolute;
  left: 14px;
  bottom: 8px;
  font-size: 0.9rem;
  color: #6b7280;
  pointer-events: none;
}
.st-key-input_console [data-testid="stTextArea"] textarea {
  min-height: var(--input-box-h) !important;
  height: var(--input-box-h) !important;
  font-size: 1.05rem !important;
}
.st-key-input_console [data-testid="stAudioInput"] {
  min-height: var(--input-box-h) !important;
}
.st-key-input_console [data-testid="stAudioInput"] button {
  min-height: var(--input-box-h) !important;
  height: var(--input-box-h) !important;
}
.st-key-input_console .stButton > button {
  height: 46px !important;
}

/* Entry landing page */
.landing-shell {
  max-width: 980px;
  margin: 6vh auto 0 auto;
  text-align: center;
}
.landing-title {
  font-size: clamp(1.8rem, 3.2vw, 2.8rem);
  font-weight: 700;
  line-height: 1.35;
  color: #1F4433;
  animation: fadePulse 2.4s ease-in-out infinite;
}
@keyframes fadePulse {
  0% { opacity: 0.35; }
  50% { opacity: 1; }
  100% { opacity: 0.35; }
}
.landing-actions {
  margin-top: 2rem;
}
.landing-intro {
  margin: 1.3rem auto 0 auto;
  max-width: 980px;
  text-align: left;
  color: #1f2937;
  line-height: 1.55;
}
.landing-cards {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.85rem;
}
.landing-info-card {
  background: #ffffff;
  border: 1px solid #c9dccc;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(31, 68, 51, 0.1);
  padding: 1rem 1.1rem;
  width: 100%;
  box-sizing: border-box;
}
.landing-section-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: #1F4433;
  margin-top: 1rem;
  margin-bottom: 0.4rem;
}
.landing-paragraph {
  font-size: 1rem;
  margin-bottom: 0.7rem;
}
.landing-bullet {
  font-size: 0.98rem;
  margin: 0.35rem 0;
}
.landing-reveal {
  opacity: 0;
  transform: translateY(18px);
  animation: landingFadeUp 0.72s ease-out forwards;
}
.landing-delay-1 { animation-delay: 0.08s; }
.landing-delay-2 { animation-delay: 0.2s; }
.landing-delay-3 { animation-delay: 0.32s; }
@keyframes landingFadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}
.st-key-landing_enter_home,
.st-key-landing_personal_info {
  opacity: 0;
  transform: translateY(18px);
  animation: landingFadeUp 0.78s ease-out forwards;
  animation-delay: 0.46s;
}
.sidebar-nav-title {
  font-size: 1rem;
  font-weight: 700;
  color: #1F4433;
  margin-top: 0.3rem;
}
.recent-condition-title {
  font-size: 0.98rem;
  font-weight: 700;
  color: #1F4433;
  margin: 0.2rem 0 0.5rem 0;
}
.recent-condition-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 0.38rem;
  margin-top: 0.2rem;
}
.recent-tag {
  display: inline-block;
  background: #f2f7f3;
  border: 1px solid #c9dccc;
  color: #1F4433;
  border-radius: 999px;
  font-size: 0.78rem;
  line-height: 1.25;
  padding: 0.28rem 0.55rem;
}

/* Chat avatar color scheme */
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
  border-radius: 999px !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: #1F4433 !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: #b05b24 !important;
}
[data-testid="stChatMessageAvatarAssistant"] svg,
[data-testid="stChatMessageAvatarUser"] svg {
  color: #ffffff !important;
  fill: #ffffff !important;
}

</style>
        """,
        unsafe_allow_html=True,
    )


def pulse_run(message: str, fn, *args, **kwargs):
    holder = st.empty()
    holder.markdown(f'<div class="pulse">{message}</div>', unsafe_allow_html=True)
    time.sleep(0.12)
    try:
        return fn(*args, **kwargs)
    finally:
        holder.empty()


# ---------- Data ----------
def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_medical_chunks() -> List[Dict[str, Any]]:
    indexed = load_json(MEDICAL_INDEX_PATH, {})
    chunks = indexed.get("chunks", []) if isinstance(indexed, dict) else []
    if chunks:
        return chunks

    raw = load_json(MEDICAL_KB_PATH, {})
    docs = raw.get("documents", []) if isinstance(raw, dict) else []
    fallback_chunks: List[Dict[str, Any]] = []
    for doc in docs:
        fallback_chunks.append(
            {
                "chunk_id": f"{doc.get('id', 'doc')}_fallback",
                "doc_id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "topic": doc.get("topic", ""),
                "subtopics": doc.get("subtopics", []),
                "issuer": doc.get("issuer", ""),
                "jurisdiction": doc.get("jurisdiction", "EU-general"),
                "updated_at": doc.get("updated_at", ""),
                "language": doc.get("language", "en"),
                "audience": doc.get("audience", []),
                "source_url": doc.get("source_url", ""),
                "version": doc.get("version", ""),
                "chunk_text": doc.get("content", ""),
                "otc_guidance": doc.get("otc_guidance", ""),
                "contraindications": doc.get("contraindications", []),
                "red_flags": doc.get("red_flags", []),
                "citation_snippet": doc.get("citation_snippet", ""),
            }
        )
    return fallback_chunks


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ---------- RAG ----------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_vector(tokens: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in tokens:
        out[t] = out.get(t, 0.0) + 1.0
    return out


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1).intersection(v2)
    num = sum(v1[k] * v2[k] for k in common)
    den1 = math.sqrt(sum(x * x for x in v1.values()))
    den2 = math.sqrt(sum(x * x for x in v2.values()))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)


def retrieve_kb_context(query: str, kb_sections: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
    qv = build_vector(tokenize(query))
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in kb_sections:
        content = f"{row.get('section', '')} {row.get('title', '')} {row.get('text', '')}"
        score = cosine_similarity(qv, build_vector(tokenize(content)))
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k] if x[0] > 0]


MEDICAL_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "fever": ["fever", "temperature", "hot"],
    "cough_cold": ["cough", "cold", "congestion", "runny nose", "sore throat"],
    "pain": ["pain", "headache", "back pain", "muscle pain", "ache"],
    "gastrointestinal": ["stomach", "nausea", "vomit", "vomiting", "diarrhea", "abdominal"],
    "hypertension": ["blood pressure", "hypertension"],
    "diabetes": ["diabetes", "blood sugar", "glucose", "hypoglycemia"],
    "urinary": ["urine", "urinary", "uti", "burning urination"],
    "constipation": ["constipation", "hard stool"],
    "sleep_anxiety": ["sleep", "insomnia", "anxiety", "stress"],
    "skin": ["rash", "itch", "hives", "skin"],
    "falls_injury": ["fall", "fell", "injury", "bruise", "fracture"],
    "medication_safety": ["interaction", "side effect", "polypharmacy", "many medicines"],
    "medication_review": ["medication review", "medicine list", "duplicate medicine"],
    "emergency_cardiac": ["chest pain", "heart pain"],
    "emergency_neuro": ["stroke", "slurred speech", "face droop", "arm weakness"],
    "respiratory_urgent": ["cannot breathe", "can't breathe", "shortness of breath", "breathing difficulty"],
}


def detect_medical_topics(user_text: str) -> Set[str]:
    t = user_text.lower()
    topics: Set[str] = set()
    for topic, kws in MEDICAL_TOPIC_KEYWORDS.items():
        if any(k in t for k in kws):
            topics.add(topic)
    return topics


def retrieve_medical_kb_context(
    query: str,
    profile: Dict[str, Any],
    medical_chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if not medical_chunks:
        return []
    qv = build_vector(tokenize(query))
    topics = detect_medical_topics(query)
    age = int(profile.get("age", 0) or 0)
    older_adult = age >= 60 or any(x in query.lower() for x in ["older", "senior", "elderly"])
    medicine_query = is_medicine_question(query)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in medical_chunks:
        content = f"{row.get('title', '')} {row.get('chunk_text', '')} {row.get('citation_snippet', '')}"
        base = cosine_similarity(qv, build_vector(tokenize(content)))
        boost = 0.0
        if topics and row.get("topic") in topics:
            boost += 0.25
        aud = set(row.get("audience", []) or [])
        if older_adult and "older_adult" in aud:
            boost += 0.18
        if medicine_query and row.get("otc_guidance"):
            boost += 0.12
        if older_adult and row.get("topic") in {"medication_safety", "medication_review"}:
            boost += 0.08
        scored.append((base + boost, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [r for s, r in scored[:top_k] if s > 0.01]
    if out:
        return out
    return [r for _, r in scored[:top_k]]


def build_medical_context_block(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No medical RAG context found."
    lines = []
    for idx, r in enumerate(rows, start=1):
        lines.append(
            f"{idx}. {r.get('title', 'Medical guidance')} | topic={r.get('topic', '')} | "
            f"issuer={r.get('issuer', '')} | updated={r.get('updated_at', '')} | url={r.get('source_url', '')}\n"
            f"   key: {r.get('citation_snippet', '')}\n"
            f"   otc: {r.get('otc_guidance', '')}\n"
            f"   red_flags: {', '.join(r.get('red_flags', [])[:4])}"
        )
    return "\n".join(lines)


CLINICAL_SYMPTOM_PATTERNS: Dict[str, List[str]] = {
    "headache": ["headache", "head pain", "migraine"],
    "fever": ["fever", "temperature", "hot"],
    "cough": ["cough", "coughing"],
    "breathing difficulty": ["can't breathe", "cannot breathe", "shortness of breath", "breathing difficulty"],
    "chest pain": ["chest pain", "chest pressure", "chest tightness"],
    "dizziness": ["dizzy", "dizziness", "lightheaded"],
    "nausea/vomiting": ["nausea", "vomit", "vomiting"],
    "abdominal pain": ["stomach pain", "abdominal pain", "belly pain"],
    "rash": ["rash", "hives", "itching"],
}

SEVERITY_PATTERNS: Dict[str, List[str]] = {
    "mild": ["mild", "slight"],
    "moderate": ["moderate", "manageable"],
    "severe": ["severe", "very bad", "intense", "unbearable"],
}

RED_FLAG_TERMS = [
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "stroke", "face droop", "slurred speech", "arm weakness",
    "unconscious", "fainted", "severe bleeding", "confusion",
]

CHRONIC_CONDITION_PATTERNS: Dict[str, List[str]] = {
    "diabetes": ["diabetes", "type 1 diabetes", "type 2 diabetes"],
    "hypertension": ["hypertension", "high blood pressure"],
    "asthma": ["asthma"],
    "copd": ["copd", "chronic obstructive pulmonary disease"],
    "kidney disease": ["kidney disease", "ckd", "renal disease"],
    "heart disease": ["heart disease", "heart failure", "coronary disease"],
    "stroke history": ["stroke history", "previous stroke", "had a stroke"],
    "cancer history": ["cancer", "oncology history"],
}

COMPLICATION_PATTERNS: Dict[str, List[str]] = {
    "dehydration risk": ["dehydrated", "not drinking", "dry mouth", "very little urine"],
    "respiratory complication risk": ["wheezing", "breath worse", "shortness of breath", "cannot breathe"],
    "neurologic warning": ["confusion", "fainting", "unconscious"],
    "infection escalation risk": ["fever getting worse", "persistent fever", "high fever"],
    "chest complication warning": ["chest pain", "chest pressure"],
}

CLINICAL_SLOTS = [
    "main symptom",
    "symptom duration",
    "measured temperature",
    "red-flag symptoms",
    "past medical history",
    "allergy information",
    "current medications",
    "age or pregnancy status",
]


def empty_clinical_context() -> Dict[str, Any]:
    return {
        "symptoms": [],
        "duration_text": "",
        "duration_days": 0,
        "temperature_c": 0.0,
        "temperature_source": "",
        "severity_text": "",
        "onset_text": "",
        "red_flags": [],
        "chronic_conditions": [],
        "complications": [],
        "history_text": "",
        "allergies_text": "",
        "medications_text": "",
        "age": 0,
        "pregnancy_status": "",
        "last_user_message": "",
        "last_updated": "",
        "slot_status": {k: "unknown" for k in CLINICAL_SLOTS},
    }


def normalize_duration_days(text: str) -> Tuple[int, str]:
    t = str(text or "").lower()
    if not t:
        return 0, ""
    direct = re.search(r"\b(\d+)\s*(hour|hours|day|days|week|weeks|month|months)\b", t)
    if direct:
        num = int(direct.group(1))
        unit = direct.group(2)
        if "hour" in unit:
            return max(1, round(num / 24)), f"{num} {unit}"
        if "day" in unit:
            return num, f"{num} {unit}"
        if "week" in unit:
            return num * 7, f"{num} {unit}"
        if "month" in unit:
            return num * 30, f"{num} {unit}"
    if "today" in t:
        return 1, "today"
    if "yesterday" in t:
        return 2, "since yesterday"
    if "this week" in t:
        return 7, "this week"
    if "two weeks" in t:
        return 14, "two weeks"
    return 0, ""


def extract_temperature_c(user_text: str) -> Tuple[float, str]:
    text = str(user_text or "").lower()
    if not text:
        return 0.0, ""

    c_match = re.search(r"\b(3[5-9](?:\.\d+)?)\s*(?:°?\s*c|celsius)\b", text)
    if c_match:
        val = float(c_match.group(1))
        return round(val, 1), "celsius"

    f_match = re.search(r"\b(9[0-9]|1[0-1][0-9]|120)(?:\.\d+)?\s*(?:°?\s*f|fahrenheit)\b", text)
    if f_match:
        f_val = float(f_match.group(1))
        c_val = (f_val - 32.0) * 5.0 / 9.0
        return round(c_val, 1), "fahrenheit"

    # Bare number fallback near fever words.
    if any(x in text for x in ["fever", "temperature", "temp"]):
        bare = re.search(r"\b(3[6-9](?:\.\d+)?)\b", text)
        if bare:
            return round(float(bare.group(1)), 1), "celsius_assumed"
    return 0.0, ""


def extract_symptoms(user_text: str) -> List[str]:
    t = str(user_text or "").lower()
    hits: List[str] = []
    for canonical, patterns in CLINICAL_SYMPTOM_PATTERNS.items():
        if any(p in t for p in patterns):
            hits.append(canonical)
    return hits


def extract_severity_text(user_text: str) -> str:
    t = str(user_text or "").lower()
    for level, patterns in SEVERITY_PATTERNS.items():
        if any(p in t for p in patterns):
            return level
    score = re.search(r"\b([0-9]|10)\s*/\s*10\b", t)
    if score:
        val = int(score.group(1))
        if val >= 8:
            return "severe"
        if val >= 5:
            return "moderate"
        return "mild"
    return ""


def extract_red_flags(user_text: str) -> List[str]:
    t = str(user_text or "").lower()
    return [flag for flag in RED_FLAG_TERMS if flag in t]


def extract_chronic_conditions(user_text: str, profile: Dict[str, Any]) -> List[str]:
    text = f"{str(user_text or '').lower()} {str(profile.get('health_notes', '')).lower()}"
    found: List[str] = []
    for label, patterns in CHRONIC_CONDITION_PATTERNS.items():
        if any(p in text for p in patterns):
            found.append(label)
    return found


def extract_complications(user_text: str) -> List[str]:
    text = str(user_text or "").lower()
    found: List[str] = []
    for label, patterns in COMPLICATION_PATTERNS.items():
        if any(p in text for p in patterns):
            found.append(label)
    return found


def summarize_clinical_context(clinical_context: Dict[str, Any]) -> str:
    symptoms = ", ".join(clinical_context.get("symptoms", []) or []) or "none provided"
    duration_text = clinical_context.get("duration_text", "") or "not provided"
    severity = clinical_context.get("severity_text", "") or "not provided"
    history = clinical_context.get("history_text", "") or "not provided"
    allergies = clinical_context.get("allergies_text", "") or "not provided"
    meds = clinical_context.get("medications_text", "") or "not provided"
    age = clinical_context.get("age", 0) or 0
    preg = clinical_context.get("pregnancy_status", "") or "not provided"
    red_flags = ", ".join(clinical_context.get("red_flags", []) or []) or "none detected"
    chronic = ", ".join(clinical_context.get("chronic_conditions", []) or []) or "none provided"
    complications = ", ".join(clinical_context.get("complications", []) or []) or "none detected"
    temp_c = float(clinical_context.get("temperature_c", 0.0) or 0.0)
    temp_text = f"{temp_c:.1f}C" if temp_c > 0 else "not provided"
    return (
        f"Symptoms: {symptoms}; Duration: {duration_text}; Severity: {severity}; "
        f"History: {history}; Allergies: {allergies}; Medications: {meds}; "
        f"Temperature: {temp_text}; Age: {age}; Pregnancy: {preg}; Red flags: {red_flags}; "
        f"Chronic conditions: {chronic}; Complications: {complications}."
    )


def update_clinical_context(clinical_context: Dict[str, Any], user_text: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    ctx = dict(clinical_context or empty_clinical_context())
    slot_status = dict(ctx.get("slot_status", {}) or {k: "unknown" for k in CLINICAL_SLOTS})
    text = str(user_text or "").strip()
    low = text.lower()
    symptoms = list(ctx.get("symptoms", []) or [])
    for s in extract_symptoms(text):
        if s not in symptoms:
            symptoms.append(s)
    ctx["symptoms"] = symptoms

    days, duration_text = normalize_duration_days(text)
    if days > 0:
        prev_days = int(ctx.get("duration_days", 0) or 0)
        if days >= prev_days:
            ctx["duration_days"] = days
            ctx["duration_text"] = duration_text or ctx.get("duration_text", "")

    temp_c, temp_src = extract_temperature_c(text)
    if temp_c > 0:
        prev_temp = float(ctx.get("temperature_c", 0.0) or 0.0)
        if temp_c >= prev_temp:
            ctx["temperature_c"] = temp_c
            ctx["temperature_source"] = temp_src

    sev = extract_severity_text(text)
    if sev:
        ctx["severity_text"] = sev
    elif float(ctx.get("temperature_c", 0.0) or 0.0) >= 39.0:
        ctx["severity_text"] = "severe"

    red_flags = list(ctx.get("red_flags", []) or [])
    for rf in extract_red_flags(text):
        if rf not in red_flags:
            red_flags.append(rf)
    ctx["red_flags"] = red_flags

    chronic_conditions = list(ctx.get("chronic_conditions", []) or [])
    for cc in extract_chronic_conditions(text, profile):
        if cc not in chronic_conditions:
            chronic_conditions.append(cc)
    ctx["chronic_conditions"] = chronic_conditions

    complications = list(ctx.get("complications", []) or [])
    for cp in extract_complications(text):
        if cp not in complications:
            complications.append(cp)
    ctx["complications"] = complications

    if red_flags:
        ctx["red_flags_answered"] = True
    elif any(x in low for x in ["no warning signs", "no chest pain", "no breathing trouble", "none of those", "no red flags"]):
        ctx["red_flags_answered"] = True

    history_text = str(profile.get("health_notes", "")).strip()
    if history_text:
        ctx["history_text"] = history_text
    elif any(x in low for x in ["history of", "chronic", "diabetes", "hypertension"]):
        ctx["history_text"] = text

    allergies_text = str(profile.get("allergies", "")).strip()
    if allergies_text:
        ctx["allergies_text"] = allergies_text
    elif "no allergies" in low or "none allergies" in low:
        ctx["allergies_text"] = "none reported"
    elif "allerg" in low:
        ctx["allergies_text"] = text

    meds_text = str(profile.get("current_medications", "")).strip()
    if meds_text:
        ctx["medications_text"] = meds_text
    elif any(x in low for x in ["i take", "taking", "my meds", "medication"]):
        ctx["medications_text"] = text
    elif "no medication" in low or "not taking any" in low:
        ctx["medications_text"] = "none reported"

    ctx["age"] = int(profile.get("age", 0) or 0)
    ctx["pregnancy_status"] = str(profile.get("pregnancy_status", "")).strip()
    ctx["last_user_message"] = text
    ctx["last_updated"] = datetime.utcnow().isoformat()
    ctx["slot_status"] = slot_status
    return ctx


def infer_intake_gaps_from_context(clinical_context: Dict[str, Any]) -> List[str]:
    ctx = clinical_context or {}
    gaps: List[str] = []
    symptoms = ctx.get("symptoms", []) or []
    if not symptoms:
        gaps.append("main symptom")
    if not str(ctx.get("duration_text", "")).strip():
        gaps.append("symptom duration")
    if "fever" in symptoms and float(ctx.get("temperature_c", 0.0) or 0.0) <= 0:
        gaps.append("measured temperature")
    if symptoms and not (ctx.get("red_flags", []) or []) and not bool(ctx.get("red_flags_answered")):
        gaps.append("red-flag symptoms")
    if not str(ctx.get("history_text", "")).strip():
        gaps.append("past medical history")
    if not str(ctx.get("allergies_text", "")).strip():
        gaps.append("allergy information")
    if not str(ctx.get("medications_text", "")).strip():
        gaps.append("current medications")
    age = int(ctx.get("age", 0) or 0)
    preg = str(ctx.get("pregnancy_status", "")).strip().lower()
    if age <= 0 and "preg" not in preg:
        gaps.append("age or pregnancy status")
    return gaps


def slot_has_value(slot: str, ctx: Dict[str, Any]) -> bool:
    c = ctx or {}
    if slot == "main symptom":
        return bool(c.get("symptoms"))
    if slot == "symptom duration":
        return bool(str(c.get("duration_text", "")).strip()) or int(c.get("duration_days", 0) or 0) > 0
    if slot == "measured temperature":
        return float(c.get("temperature_c", 0.0) or 0.0) > 0
    if slot == "red-flag symptoms":
        # explicit yes/no content can come later; list existing flags counts as known.
        return "red_flags_answered" in c or bool(c.get("red_flags"))
    if slot == "past medical history":
        return bool(str(c.get("history_text", "")).strip())
    if slot == "allergy information":
        return bool(str(c.get("allergies_text", "")).strip())
    if slot == "current medications":
        return bool(str(c.get("medications_text", "")).strip())
    if slot == "age or pregnancy status":
        return int(c.get("age", 0) or 0) > 0 or ("preg" in str(c.get("pregnancy_status", "")).lower())
    return False


def slot_value_signature(slot: str, ctx: Dict[str, Any]) -> str:
    c = ctx or {}
    if slot == "main symptom":
        return "|".join(sorted([str(x).strip().lower() for x in (c.get("symptoms", []) or []) if str(x).strip()]))
    if slot == "symptom duration":
        return str(int(c.get("duration_days", 0) or 0))
    if slot == "measured temperature":
        return str(float(c.get("temperature_c", 0.0) or 0.0))
    if slot == "red-flag symptoms":
        red = "|".join(sorted([str(x).strip().lower() for x in (c.get("red_flags", []) or []) if str(x).strip()]))
        answered = "1" if c.get("red_flags_answered") else "0"
        return f"{answered}:{red}"
    if slot == "past medical history":
        return str(c.get("history_text", "")).strip().lower()
    if slot == "allergy information":
        return str(c.get("allergies_text", "")).strip().lower()
    if slot == "current medications":
        return str(c.get("medications_text", "")).strip().lower()
    if slot == "age or pregnancy status":
        age = int(c.get("age", 0) or 0)
        preg = str(c.get("pregnancy_status", "")).strip().lower()
        return f"{age}:{preg}"
    return ""


def update_slot_status(
    previous_ctx: Dict[str, Any],
    current_ctx: Dict[str, Any],
    last_question_target: str,
    user_text: str,
) -> Dict[str, str]:
    prev_status = dict((current_ctx or {}).get("slot_status", {}) or {})
    if not prev_status:
        prev_status = {k: "unknown" for k in CLINICAL_SLOTS}
    next_status = dict(prev_status)

    for slot in CLINICAL_SLOTS:
        had = slot_has_value(slot, previous_ctx or {})
        has_now = slot_has_value(slot, current_ctx or {})
        if has_now and not had:
            next_status[slot] = "filled"
        elif has_now and had and looks_like_target_answer(user_text, slot):
            next_status[slot] = "confirmed"
        elif not has_now and next_status.get(slot) != "asked":
            next_status[slot] = "unknown"

    if last_question_target and not slot_has_value(last_question_target, current_ctx or {}):
        next_status[last_question_target] = "asked"
    return next_status


def compute_clinical_delta(
    previous_ctx: Dict[str, Any],
    current_ctx: Dict[str, Any],
    previous_triage: Optional[Dict[str, Any]] = None,
    current_triage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    newly_filled: List[str] = []
    changed_slots: List[str] = []
    for slot in CLINICAL_SLOTS:
        had = slot_has_value(slot, previous_ctx or {})
        has_now = slot_has_value(slot, current_ctx or {})
        if has_now and not had:
            newly_filled.append(slot)
        elif has_now and had:
            if slot_value_signature(slot, previous_ctx or {}) != slot_value_signature(slot, current_ctx or {}):
                changed_slots.append(slot)

    unresolved = infer_intake_gaps_from_context(current_ctx or {})
    prev_risk = risk_label_from_triage(previous_triage)
    curr_risk = risk_label_from_triage(current_triage)
    return {
        "newly_filled": newly_filled,
        "changed_slots": changed_slots,
        "still_missing": unresolved,
        "risk_changed": prev_risk != curr_risk,
        "prev_risk": prev_risk,
        "curr_risk": curr_risk,
    }


def format_delta_ack(delta: Dict[str, Any], clinical_context: Dict[str, Any]) -> str:
    return format_delta_ack_with_style(delta, clinical_context, 0)


def format_delta_ack_with_style(delta: Dict[str, Any], clinical_context: Dict[str, Any], style_index: int = 0) -> str:
    if not delta:
        return ""
    newly = list(delta.get("newly_filled", []) or [])
    changed = list(delta.get("changed_slots", []) or [])
    symptoms = ", ".join((clinical_context or {}).get("symptoms", []) or [])
    duration_text = str((clinical_context or {}).get("duration_text", "")).strip()
    severity = str((clinical_context or {}).get("severity_text", "")).strip()
    temp_c = float((clinical_context or {}).get("temperature_c", 0.0) or 0.0)
    openers = [
        "Thanks, I noted",
        "Got it, I recorded",
        "I hear you, and I captured",
        "Thank you, I updated",
        "Okay, I logged",
        "I've added",
        "I now have",
        "Understood, I captured",
        "I've registered",
        "That helps, I noted",
        "Thanks for clarifying, I logged",
        "I've updated the record with",
    ]
    opener = openers[style_index % len(openers)]

    if "symptom duration" in newly or "symptom duration" in changed:
        if symptoms and duration_text:
            return f"{opener} that your {symptoms} has lasted {duration_text}."
        if duration_text:
            return f"{opener} that the duration is {duration_text}."
    if "measured temperature" in newly or "measured temperature" in changed:
        if temp_c > 0:
            return f"{opener} that your measured temperature is {temp_c:.1f}C."
    if "severity level" in newly or "severity level" in changed:
        if severity:
            return f"{opener} that the severity is {severity}."
    if "main symptom" in newly or "main symptom" in changed:
        if symptoms:
            return f"{opener} that the main symptom is {symptoms}."
    if newly:
        first = newly[0]
        return f"{opener} your update on {first}."
    return f"{opener} your update."


def clean_response_grammar(text: str) -> str:
    out = str(text or "").replace("\r\n", "\n")
    replacements = {
        "Could you share your whether": "Could you share whether",
        "Could you share your if": "Could you share if",
        "you your": "your",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    # Preserve markdown line breaks and only normalize repeated spaces/tabs.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def reply_similarity(a: str, b: str) -> float:
    aa = re.sub(r"\s+", " ", str(a or "").strip().lower())
    bb = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(a=aa, b=bb).ratio()


def ensure_delta_mentioned(reply: str, delta_ack: str) -> str:
    if not delta_ack:
        return reply
    low = str(reply or "").lower()
    key_terms = [x for x in re.findall(r"[a-zA-Z0-9]+", delta_ack.lower()) if len(x) > 3]
    if any(k in low for k in key_terms):
        return reply
    return f"{delta_ack} {reply}".strip()


def choose_next_question_target(gaps: List[str], last_target: str = "") -> str:
    if not gaps:
        return ""
    priorities = [
        "symptom duration",
        "measured temperature",
        "red-flag symptoms",
        "current medications",
        "allergy information",
        "past medical history",
        "age or pregnancy status",
        "main symptom",
    ]
    for p in priorities:
        if p in gaps and p != last_target:
            return p
    return gaps[0]


def target_filled_between(target: str, before_ctx: Dict[str, Any], after_ctx: Dict[str, Any]) -> bool:
    if not target:
        return False
    before = before_ctx or {}
    after = after_ctx or {}
    field_map = {
        "main symptom": "symptoms",
        "symptom duration": "duration_text",
        "measured temperature": "temperature_c",
        "red-flag symptoms": "red_flags",
        "past medical history": "history_text",
        "allergy information": "allergies_text",
        "current medications": "medications_text",
        "age or pregnancy status": "age",
    }
    field = field_map.get(target)
    if not field:
        return False
    b = before.get(field)
    a = after.get(field)
    if isinstance(a, list):
        return bool(a) and len(a) >= len(b or [])
    if field == "age":
        return int(a or 0) > 0
    if field == "temperature_c":
        return float(a or 0.0) > 0
    return bool(str(a or "").strip()) and str(a).strip() != str(b or "").strip()


def looks_like_target_answer(user_text: str, target: str) -> bool:
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    if target == "symptom duration":
        return normalize_duration_days(text)[0] > 0
    if target == "measured temperature":
        return extract_temperature_c(text)[0] > 0
    if target == "red-flag symptoms":
        return any(x in text for x in ["yes", "no", "none", "chest pain", "breathing", "faint", "confusion"])
    if target == "current medications":
        return any(x in text for x in ["take", "medication", "meds", "none"]) or len(text.split()) >= 2
    if target == "allergy information":
        return "allerg" in text or "none" in text or "no allergy" in text
    if target == "past medical history":
        return any(x in text for x in ["history", "diabetes", "hypertension", "none", "chronic"])
    if target == "age or pregnancy status":
        return bool(re.search(r"\b\d{1,3}\b", text)) or "pregnan" in text
    if target == "main symptom":
        return bool(extract_symptoms(text))
    return len(text.split()) >= 2


def is_clinical_followup_turn(user_text: str, clinical_context: Dict[str, Any], last_question_target: str) -> bool:
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    if is_healthcare_or_navigation_intent(text):
        return True
    if last_question_target:
        if looks_like_target_answer(text, last_question_target):
            return True
    ctx = clinical_context or {}
    has_existing_symptom = bool(ctx.get("symptoms"))
    has_duration_like = bool(normalize_duration_days(text)[0] > 0)
    has_severity_like = bool(extract_severity_text(text))
    has_slot_like = any(
        x in text for x in ["allerg", "medication", "meds", "history of", "chronic", "warning sign", "red flag"]
    )
    if has_existing_symptom and (has_duration_like or has_severity_like or has_slot_like):
        return True
    return False


def apply_risk_floor(triage: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
    t = dict(triage or {})
    symptoms = set((clinical_context or {}).get("symptoms", []) or [])
    duration_days = int((clinical_context or {}).get("duration_days", 0) or 0)
    if "headache" in symptoms and duration_days >= 14:
        current = str(t.get("severity", "low")).lower()
        order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
        if order.get(current, 0) < order["moderate"]:
            t["severity"] = "moderate"
            t["needs_urgent_care"] = False
            t["recommended_care_type"] = t.get("recommended_care_type") or "clinic"
            summary = str(t.get("triage_summary", "")).strip()
            floor_note = "Risk floor applied: headache lasting 14+ days requires at least moderate risk and in-person clinical follow-up."
            t["triage_summary"] = f"{summary} {floor_note}".strip()
    return t


def continuity_guardrail_reply(reply: str, clinical_context: Dict[str, Any], response_mode: str) -> str:
    text = str(reply or "").strip()
    if not text:
        return text
    if response_mode not in {"clinical_assessment", "medication_guidance", "cause_explainer"}:
        return text
    symptoms = list((clinical_context or {}).get("symptoms", []) or [])
    duration_text = str((clinical_context or {}).get("duration_text", "")).strip()
    if symptoms:
        symptom_phrase = ", ".join(symptoms)
        bad_patterns = [
            "what symptoms have you been experiencing",
            "what symptoms do you have",
            "could you tell me what symptoms",
        ]
        lowered = text.lower()
        if any(p in lowered for p in bad_patterns):
            text = re.sub(
                r"(?i)could you.*?symptoms[^?]*\?",
                "",
                text,
            ).strip()
            if duration_text:
                prefix = f"You shared {symptom_phrase} for {duration_text}. "
            else:
                prefix = f"You shared {symptom_phrase}. "
            if not text.lower().startswith("you shared"):
                text = prefix + text
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------- Tool A: triage ----------
def heuristic_triage(user_text: str, legal_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    t = user_text.lower()
    high_kw = ["chest pain", "can't breathe", "cannot breathe", "stroke", "seizure", "fainted", "unconscious", "severe bleeding"]
    med_kw = ["fever", "dizzy", "vomit", "vomiting", "cough", "headache", "pain"]

    severity = "low"
    urgent = False
    recommended = "clinic"

    if any(k in t for k in high_kw):
        severity = "critical"
        urgent = True
        recommended = "hospital"
    elif any(k in t for k in med_kw):
        severity = "moderate"
        recommended = "clinic"

    if "no car" in t or "can't travel" in t or "cannot travel" in t or "30 km" in t:
        recommended = "mobile_clinic"

    top = legal_hits[0] if legal_hits else {
        "section": "EU_AI_ACT_ART_14",
        "text": "Human oversight required for high-risk healthcare recommendations.",
    }
    conf = 0.79 if severity in {"critical", "moderate"} else 0.74

    return {
        "severity": severity,
        "needs_urgent_care": urgent,
        "recommended_care_type": recommended,
        "triage_summary": "Initial triage completed from symptom description.",
        "confidence_score": conf,
        "legal_basis": top.get("text", ""),
        "legal_ref": top.get("section", "EU_AI_ACT_ART_14"),
        "fallback_action": "FLAG_FOR_HUMAN_REVIEW" if conf < LOW_CONFIDENCE_THRESHOLD else "AUTO_DECISION",
        "warm_reply": "I hear that this situation is stressful. I will help you step by step.",
    }


def triage_tool(user_text: str, kb_sections: List[Dict[str, Any]], clinical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    triage_input = user_text
    if clinical_context:
        triage_input = f"Current turn: {user_text}\nCumulative clinical context: {summarize_clinical_context(clinical_context)}"
    legal_hits = retrieve_kb_context(triage_input, kb_sections, top_k=2)
    fallback = heuristic_triage(triage_input, legal_hits)
    api_key = get_openai_api_key()
    if not api_key:
        fallback["triage_summary"] += " OpenAI key not configured; heuristic triage is being used."
        return fallback

    legal_context = "\n".join(
        f"- {x.get('section','')}: {x.get('title','')} | {x.get('text','')}" for x in legal_hits
    )
    if not legal_context:
        legal_context = "- EU_AI_ACT_ART_14: Human oversight required for high-risk healthcare recommendations."

    prompt = f"""
Perform rural tele-triage for this user statement:
{triage_input}

Policy context:
{legal_context}

Return JSON only with keys:
severity, needs_urgent_care, recommended_care_type, triage_summary, confidence_score, legal_basis, legal_ref, fallback_action, warm_reply
Constraints:
- severity in [low, moderate, high, critical]
- confidence_score 0..1
- if confidence_score < {LOW_CONFIDENCE_THRESHOLD}, fallback_action must be FLAG_FOR_HUMAN_REVIEW else AUTO_DECISION
""".strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            temperature=0.1,
        )
        payload = parse_json_object(response.output_text)
        if not payload:
            return fallback
        conf = round(float(payload.get("confidence_score", fallback["confidence_score"])), 3)
        output = {
            "severity": str(payload.get("severity", fallback["severity"])).lower(),
            "needs_urgent_care": bool(payload.get("needs_urgent_care", fallback["needs_urgent_care"])),
            "recommended_care_type": payload.get("recommended_care_type", fallback["recommended_care_type"]),
            "triage_summary": payload.get("triage_summary", fallback["triage_summary"]),
            "confidence_score": conf,
            "legal_basis": payload.get("legal_basis", fallback["legal_basis"]),
            "legal_ref": payload.get("legal_ref", fallback["legal_ref"]),
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW" if conf < LOW_CONFIDENCE_THRESHOLD else "AUTO_DECISION",
            "warm_reply": payload.get("warm_reply", fallback["warm_reply"]),
        }
        return apply_risk_floor(output, clinical_context or {})
    except Exception:
        return apply_risk_floor(fallback, clinical_context or {})


# ---------- Tool B: geo locator ----------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def expand_location_aliases(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return ""
    aliases = {
        "roma": "rome",
        "torino": "turin",
        "milano": "milan",
        "napoli": "naples",
        "sevilla": "seville",
        "lisboa": "lisbon",
        "praha": "prague",
        "wien": "vienna",
        "warszawa": "warsaw",
        "koln": "cologne",
        "munchen": "munich",
    }
    for src, dst in aliases.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    return t


def infer_country_from_text(text: str) -> str:
    t = expand_location_aliases(text)
    if not t:
        return ""
    aliases = {
        "Italy": ["italy"],
        "Spain": ["spain"],
        "France": ["france"],
        "Germany": ["germany", "deutschland"],
        "Netherlands": ["netherlands", "holland"],
        "Belgium": ["belgium"],
        "Portugal": ["portugal"],
        "Poland": ["poland"],
        "Austria": ["austria"],
        "Czechia": ["czechia", "czech republic"],
        "Ireland": ["ireland"],
        "Greece": ["greece"],
        "United Kingdom": ["united kingdom", "uk", "england", "scotland", "wales", "northern ireland", "great britain", "britain"],
    }
    for canonical, values in aliases.items():
        if any(v in t for v in values):
            return canonical
    return ""


def get_profile_country_value(profile: Dict[str, Any]) -> str:
    explicit_country = str(profile.get("country", "")).strip()
    if explicit_country:
        return explicit_country
    return infer_country_from_text(get_user_real_address(profile))


def get_profile_country_norm(profile: Dict[str, Any]) -> str:
    return normalize_text(get_profile_country_value(profile))


def is_resource_catalog_request(user_text: str) -> bool:
    t = user_text.lower()
    catalog_signals = [
        "show me all",
        "list all",
        "all the",
        "what are the options",
        "available options",
        "show clinics",
        "show hospitals",
        "show pharmacies",
    ]
    return any(s in t for s in catalog_signals)


def is_nearest_lookup_intent(user_text: str) -> bool:
    t = user_text.lower()
    nearest_signals = [
        "nearest",
        "closest",
        "how far",
        "near me",
        "nearby",
        "where is the nearest",
    ]
    return any(s in t for s in nearest_signals)


def extract_requested_resource_types(user_text: str) -> List[str]:
    t = user_text.lower()
    out: List[str] = []
    if any(k in t for k in ["pharmacy", "pharmacies", "drugstore", "drugstores"]):
        out.append("pharmacy")
    if any(k in t for k in ["clinic", "clinics"]):
        out.append("clinic")
    if any(k in t for k in ["hospital", "hospitals"]):
        out.append("hospital")
    if any(k in t for k in ["mobile clinic", "mobile clinics"]):
        out.append("mobile_clinic")
    if any(k in t for k in ["transport", "ride", "rides"]):
        out.append("transport")
    return out


def get_location_match_score(profile: Dict[str, Any], resource_row: Dict[str, Any]) -> Tuple[float, List[str], bool]:
    weights = {
        "country": 4.0,
        "state_province": 3.0,
        "city": 2.5,
        "county": 1.5,
        "locality": 2.0,
    }
    score = 0.0
    matched_fields: List[str] = []
    user_address_norm = expand_location_aliases(get_user_real_address(profile))
    resource_address_norm = expand_location_aliases(resource_row.get("address", ""))

    profile_country = get_profile_country_norm(profile)
    resource_country = expand_location_aliases(resource_row.get("country", ""))
    country_mismatch = bool(profile_country and resource_country and profile_country != resource_country)

    for field, w in weights.items():
        pv = expand_location_aliases(get_profile_field_value(profile, field))
        rv = expand_location_aliases(resource_row.get(field, ""))
        if pv and rv and (pv in rv or rv in pv):
            score += w
            matched_fields.append(field)
        elif user_address_norm and rv and rv in user_address_norm:
            score += round(w * 0.9, 2)
            matched_fields.append(f"{field}_from_address")

    if user_address_norm and resource_address_norm:
        ignore = {"street", "road", "avenue", "lane", "district", "city", "county", "state", "province", "region"}
        overlap = {
            tok for tok in set(user_address_norm.split()).intersection(set(resource_address_norm.split()))
            if len(tok) >= 4 and tok not in ignore
        }
        if overlap:
            score += min(1.2, 0.2 * len(overlap))
            matched_fields.append("address_overlap")
    return score, matched_fields, country_mismatch


def geo_locator_tool(
    village: Dict[str, Any],
    resources: List[Dict[str, Any]],
    triage: Dict[str, Any],
    user_text: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text_l = user_text.lower()
    catalog_request = is_resource_catalog_request(user_text)
    requested_types = extract_requested_resource_types(user_text)
    wants_pharmacy = any(k in text_l for k in ["pharmacy", "drugstore", "medicine", "drug", "flu"])
    wants_clinic = "clinic" in text_l
    wants_hospital = any(k in text_l for k in ["hospital", "emergency room", "emergency department"])
    transport_barrier = any(k in user_text.lower() for k in ["no car", "can't travel", "cannot travel", "30 km", "no transport"])

    preferred_types: List[str]
    if catalog_request:
        preferred_types = requested_types or ["clinic", "hospital", "pharmacy"]
    elif triage.get("needs_urgent_care"):
        preferred_types = ["hospital", "mobile_clinic"]
    elif wants_hospital:
        preferred_types = ["hospital"]
    elif wants_clinic:
        preferred_types = ["clinic"]
    elif wants_pharmacy:
        preferred_types = ["pharmacy"]
    else:
        preferred_types = [triage.get("recommended_care_type", "clinic"), "clinic", "pharmacy"]

    if transport_barrier and "mobile_clinic" not in preferred_types:
        preferred_types.insert(0, "mobile_clinic")
    if transport_barrier and "transport" not in preferred_types:
        preferred_types.append("transport")

    lat, lon = float(village["lat"]), float(village["lon"])
    ranked: List[Dict[str, Any]] = []
    profile_address = get_user_real_address(profile or {})
    profile_country_norm = get_profile_country_norm(profile or {})
    covered_countries = sorted({str(r.get("country", "")).strip() for r in resources if str(r.get("country", "")).strip()})
    country_norm_set = {normalize_text(c) for c in covered_countries}
    no_country_coverage = bool(profile_country_norm and profile_country_norm not in country_norm_set)

    for row in resources:
        row_copy = dict(row)
        rlat, rlon = float(row["location"][0]), float(row["location"][1])
        row_copy["_lat"] = rlat
        row_copy["_lon"] = rlon
        if profile:
            m_score, m_fields, country_mismatch = get_location_match_score(profile, row_copy)
        else:
            m_score, m_fields, country_mismatch = 0.0, [], False
        row_copy["location_match_score"] = round(m_score, 2)
        row_copy["location_match_fields"] = m_fields
        row_copy["country_mismatch"] = country_mismatch
        ranked.append(row_copy)

    anchor_source = "simulation_base"
    if profile_address and not no_country_coverage:
        matched = [
            r for r in ranked
            if not r.get("country_mismatch") and float(r.get("location_match_score", 0.0)) >= 3.0
        ]
        if matched:
            matched.sort(key=lambda r: float(r.get("location_match_score", 0.0)), reverse=True)
            top = matched[:5]
            total_weight = sum(float(r.get("location_match_score", 0.0)) for r in top) or 1.0
            lat = sum(float(r["_lat"]) * float(r.get("location_match_score", 0.0)) for r in top) / total_weight
            lon = sum(float(r["_lon"]) * float(r.get("location_match_score", 0.0)) for r in top) / total_weight
            anchor_source = "profile_address_match"
    elif no_country_coverage:
        anchor_source = "country_not_covered"

    for row_copy in ranked:
        d = haversine_km(lat, lon, float(row_copy["_lat"]), float(row_copy["_lon"]))
        row_copy["distance_km"] = round(d, 2)
        row_copy.pop("_lat", None)
        row_copy.pop("_lon", None)

    ranked.sort(key=lambda x: (x["country_mismatch"], -x["location_match_score"], x["distance_km"]))

    filtered = [x for x in ranked if x.get("type") in preferred_types]
    chosen = filtered[0] if filtered else ranked[0]
    catalog_candidates = filtered[:8] if filtered else ranked[:8]
    chosen = dict(chosen)
    mock_streets = ["Main St", "Market St", "River Rd", "Health Ave", "Oak Lane"]
    street = mock_streets[int(chosen.get("id", 1)) % len(mock_streets)]
    chosen["mock_address"] = chosen.get(
        "address",
        f"{100 + int(chosen.get('id', 1)) * 17} {street}, {village.get('name', 'Turin, Piedmont (Italy)')}",
    )

    origin_address = ""
    if profile:
        origin_address = get_user_real_address(profile)
    origin_label = origin_address or village.get("name", "Turin, Piedmont (Italy)")

    legal_ref = "RURAL_CARE_GUIDE_2026_2" if transport_barrier else "RURAL_CARE_GUIDE_2026_1"
    legal_basis = (
        "When transportation barriers are present, coordination should prioritize mobile clinics and community transport."
        if transport_barrier
        else "Urgency and symptom profile should guide referral to the closest appropriate care resource."
    )

    return {
        "preferred_types": preferred_types,
        "transport_barrier": transport_barrier,
        "anchor_source": anchor_source,
        "no_country_coverage": no_country_coverage,
        "covered_countries": covered_countries,
        "origin_label": origin_label,
        "origin_address": origin_address,
        "catalog_request": catalog_request,
        "requested_types": requested_types,
        "catalog_candidates": catalog_candidates,
        "nearest_resource": chosen,
        "top_candidates": (filtered[:5] if filtered else ranked[:5]),
        "legal_basis": legal_basis,
        "legal_ref": legal_ref,
        "confidence_score": 0.9,
        "fallback_action": "AUTO_DECISION",
    }


# ---------- Tool C: briefing generator ----------
def build_case_context() -> Dict[str, Any]:
    return {
        "village": st.session_state.selected_village,
        "triage": st.session_state.last_triage,
        "resource": st.session_state.last_locator,
        "clinical_context": st.session_state.clinical_context,
        "chat_summary": [m for m in st.session_state.chat_history if m.get("role") in {"user", "assistant"}][-8:],
        "document_summaries": st.session_state.document_summaries[-5:],
        "created_at": datetime.utcnow().isoformat(),
    }


def briefing_tool(case_context: Dict[str, Any], kb_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    query = json.dumps(case_context, ensure_ascii=False)
    legal_hits = retrieve_kb_context(query, kb_sections, top_k=2)
    default_basis = legal_hits[0]["text"] if legal_hits else "Human oversight is required before high-impact healthcare actions."
    default_ref = legal_hits[0]["section"] if legal_hits else "EU_AI_ACT_ART_14"

    fallback_md = f"""
## RuralLink Case Briefing

- Generated at: {case_context.get('created_at')}
- Village: {case_context.get('village', {}).get('name', 'Unknown')}
- Triage severity: {case_context.get('triage', {}).get('severity', 'unknown')}
- Urgent care flag: {case_context.get('triage', {}).get('needs_urgent_care', False)}
- Recommended resource: {case_context.get('resource', {}).get('nearest_resource', {}).get('name', 'N/A')}

### Suggested Next Actions
1. Confirm symptom timeline and current vital warning signs.
2. Contact the recommended local resource and confirm immediate availability.
3. If symptoms escalate, trigger emergency pathway after user confirmation.
4. Keep a concise handoff note for the next care provider.

### Legal Basis
- {default_basis} ({default_ref})
""".strip()

    api_key = get_openai_api_key()
    if not api_key:
        return {
            "briefing_markdown": fallback_md,
            "legal_basis": default_basis,
            "legal_ref": default_ref,
            "confidence_score": 0.76,
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW",
        }

    prompt = f"""
Create a concise professional rural healthcare coordination briefing from this case context:
{json.dumps(case_context, ensure_ascii=False)}

Return JSON only with keys:
briefing_markdown, legal_basis, legal_ref, confidence_score, fallback_action
Constraints:
- briefing_markdown should be readable by nurse/pharmacist teams.
- include a short risk summary and immediate action list.
- confidence_score 0..1; if below {LOW_CONFIDENCE_THRESHOLD}, fallback_action=FLAG_FOR_HUMAN_REVIEW else AUTO_DECISION.
""".strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            temperature=0.1,
        )
        payload = parse_json_object(response.output_text)
        if not payload:
            raise ValueError("Invalid model output")
        conf = round(float(payload.get("confidence_score", 0.8)), 3)
        return {
            "briefing_markdown": payload.get("briefing_markdown", fallback_md),
            "legal_basis": payload.get("legal_basis", default_basis),
            "legal_ref": payload.get("legal_ref", default_ref),
            "confidence_score": conf,
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW" if conf < LOW_CONFIDENCE_THRESHOLD else "AUTO_DECISION",
        }
    except Exception:
        return {
            "briefing_markdown": fallback_md,
            "legal_basis": default_basis,
            "legal_ref": default_ref,
            "confidence_score": 0.76,
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW",
        }


# ---------- multimodal intake ----------
def analyze_uploaded_document(uploaded_file, kb_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    file_name = uploaded_file.name
    mime = uploaded_file.type or "application/octet-stream"
    file_bytes = uploaded_file.getvalue()
    legal_hits = retrieve_kb_context(file_name, kb_sections, top_k=1)
    legal_basis = legal_hits[0]["text"] if legal_hits else "Verify source quality and identity consistency before using uploaded health documents."
    legal_ref = legal_hits[0]["section"] if legal_hits else "EU_AI_ACT_ART_9"

    fallback = {
        "doc_type": "Unknown",
        "key_facts": ["Document uploaded successfully.", "Manual review recommended for full interpretation."],
        "red_flags": ["No model extraction available in fallback mode."],
        "confidence_score": 0.68,
        "legal_basis": legal_basis,
        "legal_ref": legal_ref,
        "fallback_action": "FLAG_FOR_HUMAN_REVIEW",
    }

    api_key = get_openai_api_key()
    if not api_key:
        return fallback

    if not mime.startswith("image/"):
        # Keep prototype simple for non-image files
        return {
            "doc_type": "Non-image document",
            "key_facts": [f"File '{file_name}' uploaded.", "Prototype extracts full content from images; this file is queued for manual review."],
            "red_flags": [],
            "confidence_score": 0.72,
            "legal_basis": legal_basis,
            "legal_ref": legal_ref,
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW",
        }

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    prompt = f"""
You are a healthcare intake assistant. Read the uploaded document image.
Return JSON only with keys:
doc_type, key_facts, red_flags, confidence_score, legal_basis, legal_ref, fallback_action
Constraints:
- key_facts and red_flags are arrays.
- include legal basis fields.
- confidence_score 0..1; if below {LOW_CONFIDENCE_THRESHOLD}, fallback_action=FLAG_FOR_HUMAN_REVIEW else AUTO_DECISION.
""".strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                    ],
                },
            ],
            temperature=0.1,
        )
        payload = parse_json_object(response.output_text)
        if not payload:
            return fallback
        conf = round(float(payload.get("confidence_score", 0.8)), 3)
        return {
            "doc_type": payload.get("doc_type", "Image document"),
            "key_facts": payload.get("key_facts", ["Document reviewed."]),
            "red_flags": payload.get("red_flags", []),
            "confidence_score": conf,
            "legal_basis": payload.get("legal_basis", legal_basis),
            "legal_ref": payload.get("legal_ref", legal_ref),
            "fallback_action": "FLAG_FOR_HUMAN_REVIEW" if conf < LOW_CONFIDENCE_THRESHOLD else "AUTO_DECISION",
        }
    except Exception:
        return fallback


# ---------- mock external integration ----------
def mock_ehds_transmit() -> str:
    holder = st.empty()
    holder.markdown('<div class="pulse">Connecting to EHDS data interface...</div>', unsafe_allow_html=True)
    time.sleep(2.0)
    holder.empty()
    return "Status 200: Case transmitted to local emergency dispatch."


def run_agent_loop(
    user_text: str,
    village: Dict[str, Any],
    resources: List[Dict[str, Any]],
    kb_sections: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
    invoke_geo: bool = True,
    triage_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    triage = triage_override or pulse_run("Running Triage Tool...", triage_tool, user_text, kb_sections)
    locator: Dict[str, Any] = {}
    geo_rag_basis = ""
    geo_rag_ref = ""
    if invoke_geo:
        locator = pulse_run("Running Geo-Locator Tool...", geo_locator_tool, village, resources, triage, user_text, profile)
        geo_hits = retrieve_kb_context(
            f"{user_text} village:{village.get('name', '')} resource:{locator.get('nearest_resource', {}).get('type', '')}",
            kb_sections,
            top_k=1,
        )
        geo_rag_basis = (
            geo_hits[0].get("text", "")
            if geo_hits
            else "Geographic access barriers should be considered during referral routing in rural care."
        )
        geo_rag_ref = geo_hits[0].get("section", "RURAL_CARE_GUIDE_2026_2") if geo_hits else "RURAL_CARE_GUIDE_2026_2"

    nearest = locator.get("nearest_resource", {})
    response = (
        f"{triage.get('warm_reply', 'I am here with you.')}\n\n"
        f"I assessed your symptoms as **{triage.get('severity', 'unknown')}** priority. "
    )
    if invoke_geo and nearest:
        response += (
            f"The closest matching resource is **{nearest.get('name', 'N/A')}** ({nearest.get('distance_km', 'N/A')} km away). "
            f"Contact: {nearest.get('phone', 'N/A')}.\n\n"
        )
    response += "I can also prepare a pharmacist/clinician briefing for you right now."

    if triage.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
        response += "\n\nI recommend a human healthcare worker review this recommendation before any external action."

    return {
        "assistant_reply": response,
        "triage": triage,
        "locator": locator,
        "geo_rag_basis": geo_rag_basis,
        "geo_rag_ref": geo_rag_ref,
    }


def get_profile_field_value(profile: Dict[str, Any], field: str) -> str:
    if field in {"city", "locality"}:
        return str(
            profile.get("city_town_village", "")
            or profile.get("city", "")
            or profile.get("locality", "")
        ).strip()
    if field in {"state_province", "county"}:
        return str(
            profile.get("state_province_county", "")
            or profile.get("state_province", "")
            or profile.get("county", "")
        ).strip()
    return str(profile.get(field, "")).strip()


def sync_merged_location_fields(profile: Dict[str, Any]) -> None:
    city_like = get_profile_field_value(profile, "city")
    state_like = get_profile_field_value(profile, "state_province")
    profile["city_town_village"] = city_like
    profile["state_province_county"] = state_like
    if city_like:
        profile["city"] = city_like
        profile["locality"] = city_like
    if state_like:
        profile["state_province"] = state_like
        profile["county"] = state_like


def compose_full_address(profile: Dict[str, Any]) -> str:
    city_like = get_profile_field_value(profile, "city")
    state_like = get_profile_field_value(profile, "state_province")
    ordered_parts = [
        str(profile.get("street_address", "")).strip(),
        city_like,
        state_like,
        str(profile.get("postal_code", "")).strip(),
        str(profile.get("country", "")).strip(),
    ]
    clean_parts = []
    for p in ordered_parts:
        if p and p not in clean_parts:
            clean_parts.append(p)
    return ", ".join(clean_parts)


def get_user_real_address(profile: Dict[str, Any]) -> str:
    raw_address = str(profile.get("address_raw", "")).strip()
    if raw_address:
        return raw_address
    full_address = compose_full_address(profile)
    legacy_address = str(profile.get("address", "")).strip()
    return full_address or legacy_address


def infer_base_area_from_profile(profile: Dict[str, Any], villages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not villages:
        return {"name": "Turin, Piedmont (Italy)", "lat": 45.0703, "lon": 7.6869}

    user_address_norm = normalize_text(get_user_real_address(profile))
    best = villages[0]
    best_score = -1.0
    weights = {"country": 4.0, "state_province": 3.0, "city": 2.5, "county": 1.5, "locality": 2.0}

    for v in villages:
        score = 0.0
        for f, w in weights.items():
            pv = normalize_text(get_profile_field_value(profile, f))
            vv = normalize_text(v.get(f, ""))
            if pv and vv and (pv in vv or vv in pv):
                score += w
            elif user_address_norm and vv and vv in user_address_norm:
                score += round(w * 0.8, 2)
        if score > best_score:
            best_score = score
            best = v
    return best


def extract_address_candidate_from_message(user_text: str) -> str:
    text = str(user_text or "").strip()
    if not text:
        return ""

    patterns = [
        r"(?is)\b(?:my home is at|my address is|i live at|our address is|home address is|my home address is|address is|address:)\s*(.+)$",
        r"(?is)\b(?:i am located at|i'm located at|im located at)\s*(.+)$",
    ]

    candidate = ""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            break

    if not candidate:
        return ""

    # Trim trailing follow-up query in the same message.
    cut = re.split(
        r"(?i)(?:[.?!]\s+|\s+)(where is|where's|show me|can you|could you|please|find|what is|which is)\b",
        candidate,
        maxsplit=1,
    )
    candidate = cut[0].strip()
    candidate = re.sub(r"^[\"'\s:,-]+|[\"'\s]+$", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate)

    if len(candidate) < 8:
        return ""

    has_address_shape = (
        "," in candidate
        or bool(re.search(r"\b\d{3,6}\b", candidate))
        or any(w in candidate.lower() for w in ["street", "st.", "road", "rd", "avenue", "ave", "via", "piazza", "lane", "ln", "blvd"])
    )
    return candidate if has_address_shape else ""


def infer_location_fields_from_address(address: str, villages: List[Dict[str, Any]]) -> Dict[str, str]:
    address_norm = expand_location_aliases(address)
    if not address_norm or not villages:
        return {}

    best = None
    best_score = 0.0
    best_non_country_match = False
    weights = {"country": 4.0, "state_province": 3.0, "city": 2.5, "county": 1.5, "locality": 2.0}
    for v in villages:
        score = 0.0
        non_country_match = False
        for f, w in weights.items():
            vv = expand_location_aliases(v.get(f, ""))
            if vv and vv in address_norm:
                score += w
                if f != "country":
                    non_country_match = True
        if score > best_score:
            best_score = score
            best = v
            best_non_country_match = non_country_match

    if not best or best_score <= 0:
        return {}
    if not best_non_country_match:
        # Avoid writing wrong city/state when only country-level match exists.
        return {}
    return {
        "country": str(best.get("country", "")).strip(),
        "state_province": str(best.get("state_province", "")).strip(),
        "city": str(best.get("city", "")).strip(),
        "county": str(best.get("county", "")).strip(),
        "locality": str(best.get("locality", "")).strip(),
    }


def extract_basic_profile_fields_from_message(user_text: str, profile: Dict[str, Any]) -> bool:
    changed = False
    text = str(user_text or "").strip()
    if not text:
        return False

    name_match = re.search(r"(?i)\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    if name_match:
        candidate_name = name_match.group(1).strip()
        if candidate_name.lower() not in {"sick", "unwell", "fine", "okay", "ok", "located"}:
            if str(profile.get("full_name", "")).strip() != candidate_name:
                profile["full_name"] = candidate_name
                changed = True

    age_match = re.search(r"(?i)\b(?:i am|i'm)\s*(\d{1,3})\s*(?:years old|yo|y/o)?\b", text)
    if age_match:
        age_val = int(age_match.group(1))
        if 0 < age_val <= 120 and int(profile.get("age", 0) or 0) != age_val:
            profile["age"] = age_val
            changed = True

    gender_map = {
        "male": "Male",
        "man": "Male",
        "female": "Female",
        "woman": "Female",
        "non-binary": "Non-binary",
        "nonbinary": "Non-binary",
    }
    gender_match = re.search(r"(?i)\b(?:i am|i'm)\s*(male|female|man|woman|non-binary|nonbinary)\b", text)
    if gender_match:
        g = gender_map.get(gender_match.group(1).lower(), "")
        if g and str(profile.get("gender", "")).strip() != g:
            profile["gender"] = g
            changed = True

    allergy_match = re.search(r"(?i)\b(allerg(?:y|ies)\s*(?:to)?\s*[:\-]?\s*[^.?!]+)", text)
    if allergy_match and not str(profile.get("allergies", "")).strip():
        profile["allergies"] = allergy_match.group(1).strip()
        changed = True

    meds_match = re.search(r"(?i)\b(i take|i am taking|current medications?|my meds?)\b[:\-]?\s*([^.?!]+)", text)
    if meds_match and not str(profile.get("current_medications", "")).strip():
        profile["current_medications"] = meds_match.group(2).strip()
        changed = True

    if any(k in text.lower() for k in ["pregnant", "pregnancy", "breastfeeding"]):
        preg = "Pregnant/Breastfeeding"
        if str(profile.get("pregnancy_status", "")).strip() != preg:
            profile["pregnancy_status"] = preg
            changed = True
    return changed


def auto_update_profile_from_message(user_text: str, profile: Dict[str, Any], villages: List[Dict[str, Any]]) -> bool:
    changed = extract_basic_profile_fields_from_message(user_text, profile)
    candidate = extract_address_candidate_from_message(user_text)
    if not candidate:
        if changed:
            sync_merged_location_fields(profile)
            profile["address"] = compose_full_address(profile)
        return changed

    if str(profile.get("address_raw", "")).strip() != candidate:
        profile["address_raw"] = candidate
        changed = True

    # Best-effort structured extraction for form auto-fill.
    postal = re.search(r"\b\d{4,6}\b", candidate)
    if postal and str(profile.get("postal_code", "")).strip() != postal.group(0):
        profile["postal_code"] = postal.group(0)
        changed = True

    city_hint = re.search(r"\b\d{4,6}\s+([A-Za-zÀ-ÿ' -]{2,40})", candidate)
    if city_hint and not str(profile.get("city", "")).strip():
        raw_city = city_hint.group(1).strip()
        raw_city = re.sub(r"\([^)]*\)", "", raw_city).strip()
        city_norm = expand_location_aliases(raw_city)
        if city_norm:
            profile["city"] = city_norm.title()
            changed = True

    country = infer_country_from_text(candidate)
    if country and str(profile.get("country", "")).strip() != country:
        profile["country"] = country
        changed = True

    inferred = infer_location_fields_from_address(candidate, villages)
    for field, value in inferred.items():
        if value and str(profile.get(field, "")).strip() != value:
            profile[field] = value
            changed = True

    if not str(profile.get("street_address", "")).strip():
        parts = [p.strip() for p in candidate.split(",") if p.strip()]
        if parts:
            profile["street_address"] = parts[0]
            changed = True

    # Keep composed field updated for sidebar display.
    if not str(profile.get("city_town_village", "")).strip():
        combined_city = str(profile.get("city", "") or profile.get("locality", "")).strip()
        if combined_city:
            profile["city_town_village"] = combined_city
            changed = True
    if not str(profile.get("state_province_county", "")).strip():
        combined_state = str(profile.get("state_province", "") or profile.get("county", "")).strip()
        if combined_state:
            profile["state_province_county"] = combined_state
            changed = True

    sync_merged_location_fields(profile)
    composed = compose_full_address(profile)
    if composed and str(profile.get("address", "")).strip() != composed:
        profile["address"] = composed
        changed = True

    return changed


def is_healthcare_or_navigation_intent(user_text: str) -> bool:
    t = user_text.lower()
    keywords = [
        "pain", "fever", "dizzy", "vomit", "cough", "breath", "breathing", "chest",
        "medicine", "pharmacy", "clinic", "hospital", "doctor", "nurse",
        "ambulance", "emergency", "unwell", "sick", "symptom",
        "no car", "cannot travel", "can't travel", "transport", "village",
    ]
    return any(k in t for k in keywords)


def is_location_support_intent(user_text: str) -> bool:
    t = user_text.lower()
    location_verbs = [
        "where", "near", "nearest", "closest", "nearby", "distance", "how far",
        "route", "address", "find", "show", "list", "locate", "direction",
    ]
    location_targets = ["pharmacy", "drugstore", "hospital", "clinic", "medical center"]
    has_verb = any(k in t for k in location_verbs)
    has_target = any(k in t for k in location_targets)
    explicit_buy = any(
        k in t for k in ["where can i buy", "where to buy", "buy this", "buy medicine", "buy medication"]
    )
    return explicit_buy or (has_verb and has_target)


def is_medicine_question(user_text: str) -> bool:
    t = user_text.lower()
    medicine_keywords = [
        "what medicine", "what medication", "what should i take", "which medicine",
        "which medication", "otc", "over the counter", "paracetamol", "ibuprofen",
        "acetaminophen", "dose", "dosage", "for fever", "for pain", "for cough",
    ]
    return any(k in t for k in medicine_keywords)


def has_symptom_description(user_text: str) -> bool:
    t = user_text.lower()
    symptom_keywords = [
        "pain", "fever", "cough", "vomit", "nausea", "dizzy", "rash", "breath",
        "headache", "stomach", "chills", "sore throat", "fatigue", "diarrhea",
        "constipation", "urinary", "weakness", "back hurts", "i feel unwell", "i am unwell",
    ]
    return any(k in t for k in symptom_keywords)


def is_cause_explainer_intent(user_text: str) -> bool:
    t = user_text.lower()
    cause_markers = [
        "why", "what is causing", "what's causing", "what caused", "cause of",
        "reason for", "why do i have", "why am i", "what could this be",
    ]
    return has_symptom_description(user_text) and any(m in t for m in cause_markers)


def detect_response_mode(
    user_text: str,
    location_intent: bool,
    medicine_intent: bool,
    triage: Optional[Dict[str, Any]] = None,
    clinical_context: Optional[Dict[str, Any]] = None,
) -> str:
    if location_intent or is_nearest_lookup_intent(user_text) or is_resource_catalog_request(user_text):
        return "location_lookup"
    if medicine_intent:
        return "medication_guidance"
    if is_cause_explainer_intent(user_text):
        return "cause_explainer"
    if has_symptom_description(user_text) or is_healthcare_or_navigation_intent(user_text):
        return "clinical_assessment"
    if clinical_context and (clinical_context.get("symptoms") or clinical_context.get("duration_text")):
        if is_clinical_followup_turn(user_text, clinical_context, ""):
            return "clinical_assessment"
    if bool((triage or {}).get("needs_urgent_care")):
        return "clinical_assessment"
    return "casual_chat"


def has_transport_barrier(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in ["no car", "can't travel", "cannot travel", "30 km", "no transport", "no ride"])


def geo_opt_out_intent(user_text: str) -> bool:
    t = user_text.lower()
    patterns = [
        "not hospital", "not a hospital", "not pharmacy", "not a pharmacy", "no map",
        "don't show map", "do not show map", "i am asking for medicine",
        "not looking for pharmacy", "not looking for hospital", "no location needed",
    ]
    return any(p in t for p in patterns)


def should_invoke_geo_rag(user_text: str, triage: Optional[Dict[str, Any]]) -> bool:
    urgent = bool((triage or {}).get("needs_urgent_care")) or str((triage or {}).get("severity", "")).lower() in {"high", "critical"}
    if geo_opt_out_intent(user_text) and not urgent:
        return False
    if urgent and not is_location_support_intent(user_text) and not has_transport_barrier(user_text):
        # Emergency escalation should be immediate and concise; avoid default map detours.
        return False
    if is_location_support_intent(user_text):
        return True
    if has_transport_barrier(user_text):
        return True
    if urgent:
        return True
    return False


def is_age_sensitive_intent(user_text: str) -> bool:
    t = user_text.lower()
    age_keywords = ["age", "child", "kid", "baby", "pediatric", "elderly", "senior", "older adult"]
    return any(k in t for k in age_keywords)


def is_gender_sensitive_intent(user_text: str) -> bool:
    t = user_text.lower()
    gender_keywords = [
        "pregnan", "menstrual", "period", "gyne", "prostate", "breast", "male", "female", "gender",
    ]
    return any(k in t for k in gender_keywords)


def get_relevant_profile_gaps(user_text: str, profile: Dict[str, Any]) -> List[str]:
    gaps: List[str] = []
    if is_location_support_intent(user_text) and not get_user_real_address(profile):
        gaps.append("address")
    if is_location_support_intent(user_text) and not get_profile_country_norm(profile):
        gaps.append("country")

    age_val = int(profile.get("age", 0) or 0)
    if is_age_sensitive_intent(user_text) and age_val <= 0:
        gaps.append("age")

    gender_val = str(profile.get("gender", "")).strip().lower()
    if is_gender_sensitive_intent(user_text) and gender_val in {"", "prefer not to say"}:
        gaps.append("gender")
    return gaps


def is_diagnosis_or_prescription_request(user_text: str) -> bool:
    t = user_text.lower()
    patterns = [
        "diagnose me", "what diagnosis", "what do i have exactly",
        "prescribe", "write a prescription", "give me prescription",
        "exact dose", "exact dosage", "how many mg", "what mg should i take",
    ]
    return any(p in t for p in patterns)


def is_high_risk_user_group(user_text: str, profile: Dict[str, Any]) -> bool:
    age = int(profile.get("age", 0) or 0)
    if age >= 65:
        return True
    h = f"{profile.get('health_notes', '')} {user_text}".lower()
    return any(k in h for k in ["pregnan", "diabetes", "kidney", "heart failure", "anticoagulant", "stroke history"])


def detect_emergency_trigger(user_text: str, triage: Optional[Dict[str, Any]], medical_rows: List[Dict[str, Any]]) -> bool:
    t = user_text.lower()
    hard_flags = [
        "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
        "stroke", "face droop", "slurred speech", "arm weakness",
        "unconscious", "fainted", "severe bleeding", "confusion",
    ]
    if any(k in t for k in hard_flags):
        return True
    triage_sev = str((triage or {}).get("severity", "")).lower()
    if bool((triage or {}).get("needs_urgent_care")) or triage_sev in {"high", "critical"}:
        return True
    for r in medical_rows:
        flags = [str(x).lower() for x in (r.get("red_flags", []) or [])]
        if any(f in t for f in flags):
            return True
    return False


def infer_intake_gaps(user_text: str, profile: Dict[str, Any], clinical_context: Optional[Dict[str, Any]] = None) -> List[str]:
    if clinical_context:
        return infer_intake_gaps_from_context(clinical_context)

    t = user_text.lower()
    fallback_ctx = empty_clinical_context()
    fallback_ctx["symptoms"] = extract_symptoms(t)
    days, duration_text = normalize_duration_days(t)
    fallback_ctx["duration_days"] = days
    fallback_ctx["duration_text"] = duration_text
    fallback_ctx["severity_text"] = extract_severity_text(t)
    fallback_ctx["history_text"] = str(profile.get("health_notes", "")).strip()
    fallback_ctx["allergies_text"] = str(profile.get("allergies", "")).strip()
    fallback_ctx["medications_text"] = str(profile.get("current_medications", "")).strip()
    fallback_ctx["age"] = int(profile.get("age", 0) or 0)
    fallback_ctx["pregnancy_status"] = str(profile.get("pregnancy_status", "")).strip()
    return infer_intake_gaps_from_context(fallback_ctx)


def risk_label_from_triage(triage: Optional[Dict[str, Any]]) -> str:
    sev = str((triage or {}).get("severity", "")).lower()
    if bool((triage or {}).get("needs_urgent_care")) or sev in {"high", "critical"}:
        return "high risk"
    if sev == "moderate":
        return "moderate risk"
    return "lower risk"


def source_attribution(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Source: WHO/NICE/EMA-style public guidance (prototype safety baseline, updated 2026)."
    top = rows[0]
    return (
        f"Source: {top.get('title', 'Medical guidance')} "
        f"({top.get('issuer', 'trusted source')}, updated {top.get('updated_at', 'n/a')})."
    )


def format_gap_question(gaps: List[str], max_items: int = 2, last_target: str = "") -> str:
    if not gaps:
        return ""
    friendly = {
        "red-flag symptoms": "whether you have warning signs (chest pain, breathing trouble, confusion, fainting)",
        "measured temperature": "your measured temperature (for example, 38.5C or 101.3F)",
    }
    first = choose_next_question_target(gaps, last_target)
    ordered = [first] + [g for g in gaps if g != first]
    pick = ordered[:max_items]
    pick_text = [friendly.get(p, p) for p in pick]
    def qfrag(txt: str) -> str:
        s = str(txt).strip()
        if s.startswith("whether "):
            return s
        if s.startswith("your "):
            return s
        return f"your {s}"
    if len(pick) == 1:
        return f"Could you share {qfrag(pick_text[0])} so I can tailor this safely?"
    return f"Could you share {qfrag(pick_text[0])} and {qfrag(pick_text[1])} so I can tailor this safely?"


def strip_source_prefix(source_note: str) -> str:
    note = str(source_note or "").strip()
    if note.lower().startswith("source:"):
        return note.split(":", 1)[1].strip()
    return note


def build_structured_health_reply(
    opening: str,
    risk_level: str,
    guidance: List[str],
    safety: List[str],
    source_note: str,
    next_step: str,
) -> str:
    guidance_items = [str(g).strip() for g in guidance if str(g).strip()]
    safety_items = [str(s).strip() for s in safety if str(s).strip()]
    guidance_lines = "\n".join([f"- {x}" for x in guidance_items])
    safety_lines = "\n".join([f"- {x}" for x in safety_items])
    source_line = strip_source_prefix(source_note) or "WHO/NICE/EMA-style public guidance (prototype baseline, updated 2026)."
    blocks = [
        str(opening or "").strip(),
        f"**Risk level:** {risk_level}",
        f"**What you can do now**\n{guidance_lines}" if guidance_lines else "",
        f"**Safety checks**\n{safety_lines}" if safety_lines else "",
        f"**Source**\n{source_line}",
        f"**Next step**\n{str(next_step or '').strip()}" if str(next_step or "").strip() else "",
    ]
    return "\n\n".join([b for b in blocks if b]).strip()


def known_clinical_facts_line(clinical_context: Optional[Dict[str, Any]], style_index: int = 0) -> str:
    ctx = clinical_context or {}
    symptoms = list(ctx.get("symptoms", []) or [])
    duration_text = str(ctx.get("duration_text", "")).strip()
    severity = str(ctx.get("severity_text", "")).strip()
    temp_c = float(ctx.get("temperature_c", 0.0) or 0.0)
    parts: List[str] = []
    if symptoms:
        parts.append(f"you mentioned {', '.join(symptoms)}")
    if duration_text:
        parts.append(f"for {duration_text}")
    if severity:
        parts.append(f"with {severity} intensity")
    if temp_c > 0:
        parts.append(f"and measured temperature {temp_c:.1f}C")
    if not parts:
        return ""
    fact_text = " ".join(parts)
    templates = [
        "Thank you for sharing; I heard that {facts}.",
        "I understand that {facts}.",
        "I've noted that {facts}.",
        "From what you told me, {facts}.",
        "I'm following you: {facts}.",
        "To make sure I got this right, {facts}.",
        "I hear you, and I captured that {facts}.",
        "So far, I have that {facts}.",
        "I appreciate the detail; {facts}.",
        "I've updated your context: {facts}.",
        "What I'm hearing is {facts}.",
        "I'm with you; you reported {facts}.",
    ]
    return templates[style_index % len(templates)].format(facts=fact_text)


def conversation_bridge(style_index: int = 0, mode: str = "clinical_assessment") -> str:
    bank = {
        "clinical_assessment": [
            "We can sort this out step by step.",
            "Let's focus on the most useful details first.",
            "I'll keep this practical and focused for you.",
            "I can help you narrow this down safely.",
            "We'll go one clear step at a time.",
            "I'll keep this simple and actionable.",
            "Let's work from the key safety checks first.",
            "I can guide this in a calm, structured way.",
            "We can build the picture with a few concrete details.",
            "I'll help you decide the safest next move.",
        ],
        "medication_guidance": [
            "I can start with safe general medicine guidance.",
            "I'll focus on practical OTC safety first.",
            "Let's begin with cautious medication basics.",
            "I can help with non-prescription options and safety checks.",
            "I'll keep this centered on safe medicine use.",
            "We can review what is usually safe to try first.",
            "I can give short OTC guidance before anything else.",
            "Let's start from interaction and dose safety.",
            "I'll keep medication advice conservative and practical.",
            "I can guide this in a pharmacist-safe way.",
        ],
        "diagnosis_boundary": [
            "I can still help you plan a safe next step.",
            "I can offer safety-focused guidance without replacing a clinician.",
            "I can support with practical triage and preparation.",
            "I can help you frame the right question for in-person care.",
            "I can guide this carefully within safe limits.",
            "I can help with education and what to do next.",
            "I can support your decision process safely.",
            "I can help you move forward with lower risk.",
            "I can provide focused guidance, even without diagnosing.",
            "I can help you prepare for pharmacist or clinician follow-up.",
        ],
    }
    options = bank.get(mode, bank["clinical_assessment"])
    return options[style_index % len(options)]


def compose_opening_line(
    facts_line: str,
    style_index: int,
    mode: str = "clinical_assessment",
    prefix: str = "",
) -> str:
    parts = [str(facts_line or "").strip(), str(prefix or "").strip(), conversation_bridge(style_index, mode)]
    return " ".join([p for p in parts if p]).strip()


def risk_shift_sentence(delta: Dict[str, Any]) -> str:
    if not delta:
        return ""
    if not delta.get("risk_changed"):
        return f"Your risk level stays {str(delta.get('curr_risk', 'lower risk')).lower()} right now."
    return (
        f"With this update, your risk level moved from "
        f"{str(delta.get('prev_risk', 'lower risk')).lower()} to {str(delta.get('curr_risk', 'lower risk')).lower()}."
    )


def build_incremental_followup_reply(
    response_mode: str,
    delta_ack: str,
    delta: Dict[str, Any],
    risk_label: str,
    next_question: str,
    source_note: str,
    medical_rows: List[Dict[str, Any]],
) -> str:
    top = medical_rows[0] if medical_rows else {}
    risk_line = risk_shift_sentence(delta) or f"Your risk level is {risk_label.lower()}."
    if response_mode == "medication_guidance":
        action = top.get("otc_guidance") or "Use general OTC care cautiously and avoid combining medicines without checking interactions."
    else:
        action = top.get("citation_snippet") or "Keep monitoring symptoms and consider in-person care if this does not improve."

    follow_q = next_question or "Would you like me to ask one quick safety check next?"
    text = (
        f"{delta_ack} {risk_line} {action} {follow_q} "
        f"{source_note}"
    ).strip()
    return clean_response_grammar(text)


def should_attach_geo_map(
    user_text: str,
    triage: Optional[Dict[str, Any]],
    locator: Optional[Dict[str, Any]],
    assistant_msg: str,
) -> bool:
    if not locator or not locator.get("nearest_resource"):
        return False
    if is_nearest_lookup_intent(user_text) or is_resource_catalog_request(user_text):
        return True
    if is_location_support_intent(user_text):
        return True
    if triage and triage.get("needs_urgent_care"):
        return True
    if locator and locator.get("transport_barrier"):
        return True
    text = (assistant_msg or "").lower()
    if "nearest option i found" in text or "i found several options for you" in text:
        return True
    return False


def generate_natural_chat_reply(
    user_text: str,
    profile: Dict[str, Any],
    triage: Optional[Dict[str, Any]] = None,
    locator: Optional[Dict[str, Any]] = None,
    profile_gaps: Optional[List[str]] = None,
    medical_chunks: Optional[List[Dict[str, Any]]] = None,
    clinical_context: Optional[Dict[str, Any]] = None,
    last_question_target: str = "",
    clinical_delta: Optional[Dict[str, Any]] = None,
    followup_turn: bool = False,
    previous_assistant_text: str = "",
    opening_style_index: int = 0,
) -> str:
    api_key = get_openai_api_key()
    fallback_base = (
        "Thanks for sharing that. I am here with you, and we can go one step at a time."
    )
    profile_gaps = profile_gaps or []
    location_intent = is_location_support_intent(user_text)
    gap_questions = {
        "address": "current address so I can tailor nearby options to your real location",
        "country": "country so I can match local healthcare resources",
        "age": "age so I can tailor recommendations safely",
        "gender": "sex or gender only if you are comfortable sharing, because it can matter for this question",
    }
    user_address = get_user_real_address(profile)
    catalog_request = is_resource_catalog_request(user_text)
    nearest_lookup = is_nearest_lookup_intent(user_text)
    medicine_intent = is_medicine_question(user_text)
    urgent_triage = bool((triage or {}).get("needs_urgent_care")) or str((triage or {}).get("severity", "")).lower() in {"high", "critical"}
    medical_rows = retrieve_medical_kb_context(user_text, profile, medical_chunks or [], top_k=3)
    medical_context = build_medical_context_block(medical_rows)
    source_note = source_attribution(medical_rows)
    boundary_rows = [r for r in medical_rows if r.get("topic") in {"medication_safety", "medication_review", "pain", "fever"}]
    boundary_source_note = source_attribution(boundary_rows or medical_rows)
    intake_gaps = infer_intake_gaps(user_text, profile, clinical_context=clinical_context)
    risk_label = risk_label_from_triage(triage)
    emergency_now = detect_emergency_trigger(user_text, triage, medical_rows)
    facts_line = known_clinical_facts_line(clinical_context, opening_style_index)
    clinical_delta = clinical_delta or {}
    delta_ack = format_delta_ack_with_style(clinical_delta, clinical_context or {}, opening_style_index)
    response_mode = detect_response_mode(
        user_text=user_text,
        location_intent=location_intent,
        medicine_intent=medicine_intent,
        triage=triage,
        clinical_context=clinical_context,
    )

    if emergency_now and not location_intent:
        text = (
            f"{facts_line + ' ' if facts_line else ''}This sounds high risk and could be an emergency. Please contact local emergency services now, "
            "or ask someone near you to call immediately. I do not want to delay urgent care with a long chat. "
            f"{source_note}"
        )
        return clean_response_grammar(text)

    if is_diagnosis_or_prescription_request(user_text):
        high_risk_group = is_high_risk_user_group(user_text, profile)
        gap_q = format_gap_question(intake_gaps, max_items=1, last_target=last_question_target) or "Would you like help preparing a clear message for a pharmacist or clinician?"
        if followup_turn:
            compact = build_incremental_followup_reply(
                response_mode="clinical_assessment",
                delta_ack=delta_ack or "Thanks for the update.",
                delta=clinical_delta,
                risk_label=risk_label,
                next_question=gap_q,
                source_note=boundary_source_note,
                medical_rows=boundary_rows or medical_rows,
            )
            compact = ensure_delta_mentioned(compact, delta_ack)
            if reply_similarity(compact, previous_assistant_text) > 0.72:
                compact = f"{delta_ack or 'Thanks for the update.'} {gap_q}"
            return clean_response_grammar(compact)
        if high_risk_group:
            return build_structured_health_reply(
                opening=compose_opening_line(
                    facts_line,
                    opening_style_index,
                    mode="diagnosis_boundary",
                    prefix="I cannot provide diagnosis, prescriptions, or patient-specific dosing in higher-risk situations.",
                ),
                risk_level=risk_label.title(),
                guidance=[
                    "Use this chat for general safety education and triage, not for final medication decisions.",
                    "Contact a pharmacist or clinician for patient-specific advice.",
                ],
                safety=[
                    "Do not start or change medicines based only on chat advice.",
                    "If symptoms are getting worse, seek in-person care promptly.",
                ],
                source_note=boundary_source_note,
                next_step=gap_q,
            )
        return build_structured_health_reply(
            opening=compose_opening_line(
                facts_line,
                opening_style_index,
                mode="diagnosis_boundary",
                prefix="I cannot replace a clinician for final diagnosis or prescription-level dosing.",
            ),
            risk_level=risk_label.title(),
            guidance=[
                "Use this guidance to prepare for a pharmacist or clinician conversation.",
                "Monitor symptom changes and keep notes on timing and severity.",
            ],
            safety=[
                "Avoid self-adjusting doses without professional confirmation.",
                "Seek urgent care if red-flag symptoms appear.",
            ],
            source_note=boundary_source_note,
            next_step=gap_q,
        )

    if response_mode == "medication_guidance" and not location_intent and not urgent_triage:
        if medical_rows:
            top = medical_rows[0]
            guidance = top.get("otc_guidance") or "OTC medicine depends on your exact symptoms and medical history."
            contraindications = ", ".join(top.get("contraindications", [])[:3])
        else:
            guidance = "OTC medicine depends on your exact symptoms, age, and medical history."
            contraindications = ""
        gap_q = format_gap_question(intake_gaps, max_items=1, last_target=last_question_target)
        safety_items = [
            contraindications or "Check interactions with your current medications and allergies.",
            "Because I am an AI assistant, please confirm any medicine choice with a pharmacist before taking something new.",
        ]
        if followup_turn:
            compact = build_incremental_followup_reply(
                response_mode=response_mode,
                delta_ack=delta_ack or "Thanks for the update.",
                delta=clinical_delta,
                risk_label=risk_label,
                next_question=(gap_q if gap_q else "Would you like a short pharmacy safety checklist?"),
                source_note=source_note,
                medical_rows=medical_rows,
            )
            compact = ensure_delta_mentioned(compact, delta_ack)
            if reply_similarity(compact, previous_assistant_text) > 0.72:
                compact = (
                    f"{delta_ack or 'Thanks for the update.'} {risk_shift_sentence(clinical_delta)} "
                    f"{gap_q if gap_q else 'Would you like one quick safety checklist question?'}"
                )
            return clean_response_grammar(compact)
        return build_structured_health_reply(
            opening=compose_opening_line(
                facts_line,
                opening_style_index,
                mode="medication_guidance",
            ),
            risk_level=risk_label.title(),
            guidance=[
                guidance,
                "Use the lowest effective OTC option for the shortest needed period.",
            ],
            safety=safety_items,
            source_note=source_note,
            next_step=(
                gap_q
                if gap_q
                else "Would you like a short checklist to use when speaking with a pharmacist?"
            ),
        )

    if response_mode == "clinical_assessment" and not location_intent:
        top = medical_rows[0] if medical_rows else {}
        explanation = top.get("citation_snippet", "I can help you with a cautious symptom-first plan.")
        safety = ", ".join(top.get("contraindications", [])[:2]) if top else "Avoid adding new medicines without checking interactions."
        gap_q = format_gap_question(intake_gaps, max_items=1, last_target=last_question_target)
        if followup_turn:
            compact = build_incremental_followup_reply(
                response_mode=response_mode,
                delta_ack=delta_ack or "Thanks for sharing that.",
                delta=clinical_delta,
                risk_label=risk_label,
                next_question=(gap_q if gap_q else "Would you like one quick safety check question?"),
                source_note=source_note,
                medical_rows=medical_rows,
            )
            compact = ensure_delta_mentioned(compact, delta_ack)
            if reply_similarity(compact, previous_assistant_text) > 0.72:
                compact = (
                    f"{delta_ack or 'Thanks for sharing that.'} {risk_shift_sentence(clinical_delta)} "
                    f"{gap_q if gap_q else 'Would you like one quick safety check question?'}"
                )
            return clean_response_grammar(compact)
        return build_structured_health_reply(
            opening=compose_opening_line(
                facts_line,
                opening_style_index,
                mode="clinical_assessment",
            ),
            risk_level=risk_label.title(),
            guidance=[
                explanation,
                "Track symptom timing, severity changes, and any triggers over the next 24 hours.",
            ],
            safety=[
                safety,
                "If symptoms worsen or new warning signs appear, seek in-person care promptly.",
            ],
            source_note=source_note,
            next_step=(
                gap_q
                if gap_q
                else "Would you like me to suggest what to monitor over the next few hours?"
            ),
        )

    if response_mode == "cause_explainer" and not location_intent:
        top = medical_rows[0] if medical_rows else {}
        explanation = top.get("citation_snippet", "Several causes are possible, and context matters.")
        gap_q = format_gap_question(intake_gaps, max_items=1, last_target=last_question_target)
        if followup_turn:
            compact = (
                f"{delta_ack or 'Thanks for the update.'} There can be a few possible reasons for this. "
                f"{risk_shift_sentence(clinical_delta)} {explanation} "
                f"{gap_q if gap_q else 'Would you like me to narrow likely causes with one more detail?'} "
                f"{source_note}"
            )
            compact = ensure_delta_mentioned(compact, delta_ack)
            if reply_similarity(compact, previous_assistant_text) > 0.72:
                compact = (
                    f"{delta_ack or 'Thanks for the update.'} "
                    f"{gap_q if gap_q else 'Would you like me to narrow likely causes with one more detail?'}"
                )
            return clean_response_grammar(compact)
        opener = compose_opening_line(facts_line, opening_style_index, mode="clinical_assessment")
        return (
            f"{opener} There can be a few possible reasons for this, and your details matter. {explanation} "
            f"{gap_q if gap_q else 'If you want, I can narrow this down based on your timeline and current medications.'} "
            f"{source_note}"
        )

    if not api_key:
        if profile_gaps:
            first_gap = profile_gaps[0]
            return f"{fallback_base} To personalize this answer, could you share your {gap_questions.get(first_gap, first_gap)}?"

        if triage and locator:
            if locator.get("no_country_coverage"):
                country_val = get_profile_country_value(profile) or "your country"
                covered = ", ".join(locator.get("covered_countries", [])[:10]) or "our current EU prototype set"
                return (
                    f"I can see your address in {country_val}, but this prototype dataset does not currently include "
                    f"resources for that country. Right now I can search in: {covered}. "
                    "Would you like me to use the nearest supported country for now?"
                )
            if catalog_request or locator.get("catalog_request"):
                items = locator.get("catalog_candidates", [])[:6]
                if items:
                    lines = []
                    for i, r in enumerate(items, start=1):
                        lines.append(
                            f"{i}) {r.get('name', 'Resource')} ({r.get('type', 'care')}), "
                            f"{r.get('city', 'Unknown city')}, {r.get('distance_km', 'N/A')} km"
                        )
                    return (
                        f"{fallback_base} I found several options near {user_address or locator.get('origin_label', 'your area')}:\n"
                        + "\n".join(lines)
                        + "\nWould you like me to filter this list to only open places?"
                    )
            nearest = locator.get("nearest_resource", {})
            nearest_addr = nearest.get("mock_address", "a nearby local address")
            if triage.get("needs_urgent_care"):
                if location_intent and user_address:
                    return (
                        "Thank you for telling me this. Based on what you shared, I want us to treat this as urgent. "
                        f"Using your address ({user_address}) as your location context, the nearest option I found is {nearest.get('name', 'a nearby care center')} "
                        f"at {nearest_addr} ({nearest.get('distance_km', 'N/A')} km away). Would you like me to help you contact them now?"
                    )
                return (
                    "Thank you for telling me this. Based on what you shared, I want us to treat this as urgent. "
                    f"The nearest option I found is {nearest.get('name', 'a nearby care center')} "
                    f"({nearest.get('distance_km', 'N/A')} km away). Would you like me to help you contact them now?"
                )
            if location_intent and user_address:
                return (
                    f"{fallback_base} Using your address ({user_address}) as your location context, I found a nearby option: "
                    f"{nearest.get('name', 'a local care resource')} at {nearest_addr} "
                    f"({nearest.get('distance_km', 'N/A')} km away). Do you want practical next steps now?"
                )
            return (
                f"{fallback_base} I found a nearby option: {nearest.get('name', 'a local care resource')} "
                f"({nearest.get('distance_km', 'N/A')} km away). Do you want practical next steps now?"
            )
        return (
            f"{fallback_base} If you want, you can tell me your symptoms, timeline, and travel situation, "
            "and I will help you plan next steps."
        )

    if location_intent and locator and locator.get("nearest_resource"):
        if profile_gaps:
            first_gap = profile_gaps[0]
            return (
                f"I can absolutely find options near you. Could you share your "
                f"{gap_questions.get(first_gap, first_gap)} first?"
            )
        if locator.get("no_country_coverage"):
            country_val = get_profile_country_value(profile) or "your country"
            covered = ", ".join(locator.get("covered_countries", [])[:10]) or "our current EU prototype set"
            return (
                f"Thanks for sharing your address in {country_val}. I can see your location context, "
                "but this prototype dataset does not currently include healthcare resources for that country yet. "
                f"Right now I can search in: {covered}. Would you like me to use the nearest supported country for now?"
            )
        if catalog_request or locator.get("catalog_request"):
            items = locator.get("catalog_candidates", [])[:6]
            if not items:
                return (
                    "I checked your area but I could not find matching resources for that request yet. "
                    "Would you like me to broaden the search by nearby cities?"
                )
            lines = []
            for i, r in enumerate(items, start=1):
                lines.append(
                    f"{i}) {r.get('name', 'Resource')} ({r.get('type', 'care')}), "
                    f"{r.get('city', 'Unknown city')}, {r.get('distance_km', 'N/A')} km"
                )
            joined = "\n".join(lines)
            return (
                f"I found several options for you near {user_address or locator.get('origin_label', 'your area')}.\n"
                f"{joined}\n"
                "Would you like me to filter this to only open facilities right now?"
            )
        if not nearest_lookup:
            # Avoid forcing nearest-resource responses for general healthcare questions.
            nearest_lookup = False
        if not nearest_lookup:
            # Let the model continue with a natural conversation path below.
            pass
        else:
            nearest = locator.get("nearest_resource", {})
            nearest_name = nearest.get("name", "a nearby care center")
            nearest_addr = nearest.get("mock_address", nearest.get("address", "address not available"))
            nearest_km = nearest.get("distance_km", "N/A")
            allowed_types = set(locator.get("preferred_types", [])) or {"pharmacy", "clinic", "hospital", "mobile_clinic"}
            filtered_top = [r for r in locator.get("top_candidates", []) if r.get("type") in allowed_types]
            nearest_country = str(nearest.get("country", "")).strip()
            try:
                nearest_km_f = float(nearest_km)
            except (TypeError, ValueError):
                nearest_km_f = 5.0
            max_second_km = max(25.0, nearest_km_f * 3.5)
            second = None
            for cand in filtered_top[1:]:
                try:
                    cand_km = float(cand.get("distance_km", 9999))
                except (TypeError, ValueError):
                    cand_km = 9999.0
                same_country = (not nearest_country) or (str(cand.get("country", "")).strip() == nearest_country)
                if same_country and cand_km <= max_second_km:
                    second = cand
                    break
            if second:
                second_name = second.get("name", "another nearby option")
                second_addr = second.get("mock_address", second.get("address", "address not available"))
                second_km = second.get("distance_km", "N/A")
                return (
                    f"Based on your location context ({user_address or locator.get('origin_label', 'your area')}), "
                    f"the nearest option I found is {nearest_name} at {nearest_addr}, about {nearest_km} km away. "
                    f"Another nearby option is {second_name} at {second_addr}, around {second_km} km away. "
                    "Would you like me to narrow this down to only clinics or only pharmacies?"
                )
            return (
                f"Based on your location context ({user_address or locator.get('origin_label', 'your area')}), "
                f"the nearest option I found is {nearest_name} at {nearest_addr}, about {nearest_km} km away. "
                "Would you like the contact details and fastest route guidance next?"
            )

    context = {
        "profile": {
            "name": profile.get("full_name", ""),
            "age": profile.get("age", ""),
            "gender": profile.get("gender", ""),
            "pregnancy_status": profile.get("pregnancy_status", ""),
            "address": user_address,
            "phone": profile.get("phone", ""),
            "allergies": profile.get("allergies", ""),
            "current_medications": profile.get("current_medications", ""),
            "village": profile.get("village", ""),
            "health_notes": profile.get("health_notes", ""),
        },
        "triage": triage or {},
        "locator": locator or {},
        "intent_flags": {
            "location_intent": location_intent,
            "age_sensitive_intent": is_age_sensitive_intent(user_text),
            "gender_sensitive_intent": is_gender_sensitive_intent(user_text),
            "medicine_intent": medicine_intent,
            "response_mode": response_mode,
        },
        "relevant_missing_profile_fields": profile_gaps,
        "medical_rag_context": medical_context,
        "clinical_context": clinical_context or {},
        "last_question_target": last_question_target,
    }
    prompt = f"""
User message:
{user_text}

Context:
{json.dumps(context, ensure_ascii=False)}

Write a natural conversational reply, warm and practical.
Important style rules:
- Default style: natural conversation with no rigid template.
- If response_mode is clinical_assessment or medication_guidance, you may use short markdown labels for clarity.
- Stay within boundary: do not provide final diagnosis, prescription decisions, or patient-specific dosing.
- Use personal info (address, age, gender) only when relevant to this message.
- For clinical messages, reflect short intake framing (symptom, duration, severity, history, allergies, current meds, age/pregnancy) and ask only essential missing items.
- Start with risk stratification in plain language before recommendations.
- For medicine questions, provide general OTC education first and include a pharmacist safety confirmation note.
- For location questions, use the exact user address from context when available as location context.
- The resource address may be a mock address from locator.nearest_resource.mock_address.
- If relevant info is missing in relevant_missing_profile_fields, ask for it in one friendly question.
- Do not ask again for information already present in clinical_context.
- If clinical_context already includes the main symptom, do not ask "what symptom do you have" again.
- Do NOT always mention severity level or nearest resource.
- Do not introduce location/map suggestions unless the user explicitly asks for nearby options or context indicates urgent or transport barriers.
- When medical_rag_context is present, ground your advice in it and cite one source title + updated date naturally in the sentence.
- If emergency red-flag patterns are present, give immediate emergency escalation advice and keep the response brief.
- Mention urgency/resources only when clearly helpful for this specific message.
- End with one simple follow-up question.
""".strip()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            temperature=0.45,
        )
        text = (response.output_text or "").strip()
        return text if text else fallback_base
    except Exception:
        return fallback_base


def resolve_village(villages: List[Dict[str, Any]], village_name: str) -> Dict[str, Any]:
    for v in villages:
        if v.get("name") == village_name:
            return v
    return {"name": village_name or "Turin, Piedmont (Italy)", "lat": 45.0703, "lon": 7.6869}


def parse_chat_payload(payload: Any) -> Tuple[str, List[Any], Optional[Any]]:
    if payload is None:
        return "", [], None
    if isinstance(payload, str):
        return payload.strip(), [], None
    text = getattr(payload, "text", "")
    files = list(getattr(payload, "files", []) or [])
    audio = getattr(payload, "audio", None)
    if not files and isinstance(payload, dict):
        text = payload.get("text", text)
        files = list(payload.get("files", []))
        audio = payload.get("audio", audio)
    return str(text).strip(), files, audio


def transcribe_audio_file(file_obj: Any) -> str:
    api_key = get_openai_api_key()
    if not api_key:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        file_name = str(getattr(file_obj, "name", "") or "voice_input.wav")
        raw = file_obj.getvalue() if hasattr(file_obj, "getvalue") else b""
        if not raw:
            return ""
        audio_file = io.BytesIO(raw)
        audio_file.name = file_name
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        text = getattr(transcript, "text", "") or ""
        return str(text).strip()
    except Exception:
        return ""


def process_uploaded_files_in_chat(uploaded_files: List[Any], kb_sections: List[Dict[str, Any]]) -> int:
    handled = 0
    for f in uploaded_files:
        result = pulse_run("Analyzing uploaded document...", analyze_uploaded_document, f, kb_sections)
        st.session_state.document_summaries.append(
            {
                "filename": f.name,
                "analysis": result,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        msg = (
            f"I reviewed **{f.name}** ({result.get('doc_type')}).\n"
            f"Key facts: {', '.join(result.get('key_facts', []))}"
        )
        if result.get("red_flags"):
            msg += "\nPotential issues: " + "; ".join(result.get("red_flags", []))
        if result.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
            msg += "\nI recommend human review due to low confidence or limited extraction."
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        handled += 1
    return handled


def build_geo_map_snapshot(
    village: Dict[str, Any],
    locator: Dict[str, Any],
    resources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_candidates: List[Dict[str, Any]] = []
    if locator.get("top_candidates"):
        top_candidates = list(locator["top_candidates"])[:5]
    else:
        fallback = []
        for r in resources[:8]:
            row = dict(r)
            row["distance_km"] = round(
                haversine_km(
                    float(village["lat"]),
                    float(village["lon"]),
                    float(r["location"][0]),
                    float(r["location"][1]),
                ),
                2,
            )
            fallback.append(row)
        top_candidates = sorted(fallback, key=lambda x: x["distance_km"])[:5]

    return {
        "origin_label": locator.get("origin_label") or village.get("name", "Patient area"),
        "origin_lat": float(village.get("lat", 45.0703)),
        "origin_lon": float(village.get("lon", 7.6869)),
        "top_candidates": top_candidates,
    }


def render_geo_rag_panel_snapshot(snapshot: Dict[str, Any]) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Geo-RAG in This Conversation**")

    top_candidates = list(snapshot.get("top_candidates", []))
    origin_label = str(snapshot.get("origin_label", "Patient area"))
    origin_lat = float(snapshot.get("origin_lat", 45.0703))
    origin_lon = float(snapshot.get("origin_lon", 7.6869))

    rows = [{"label": f"{origin_label} (Patient)", "lat": origin_lat, "lon": origin_lon}]
    for r in top_candidates:
        rows.append({"label": r["name"], "lat": r["location"][0], "lon": r["location"][1]})
    map_df = pd.DataFrame(rows)
    st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]], zoom=10)
    if top_candidates:
        st.dataframe(
            pd.DataFrame(top_candidates)[["name", "type", "status", "phone", "distance_km"]],
            use_container_width=True,
        )
    else:
        st.caption("No mapped resource candidates for this turn.")
    st.markdown("</div>", unsafe_allow_html=True)


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello, I am RuralLink Agent. Tell me what you are feeling and any travel barriers. "
                    "I will coordinate triage, nearby options, and a practical care plan."
                ),
            }
        ]
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = []
    if "last_triage" not in st.session_state:
        st.session_state.last_triage = {}
    if "last_locator" not in st.session_state:
        st.session_state.last_locator = {}
    if "last_geo_rag_basis" not in st.session_state:
        st.session_state.last_geo_rag_basis = ""
    if "last_geo_rag_ref" not in st.session_state:
        st.session_state.last_geo_rag_ref = ""
    if "last_briefing" not in st.session_state:
        st.session_state.last_briefing = ""
    if "dispatch_status" not in st.session_state:
        st.session_state.dispatch_status = ""
    if "clinical_context" not in st.session_state:
        st.session_state.clinical_context = empty_clinical_context()
    else:
        cc = dict(st.session_state.clinical_context or {})
        cc.setdefault("slot_status", {k: "unknown" for k in CLINICAL_SLOTS})
        cc.setdefault("chronic_conditions", [])
        cc.setdefault("complications", [])
        st.session_state.clinical_context = cc
    if "last_question_target" not in st.session_state:
        st.session_state.last_question_target = ""
    if "opening_style_index" not in st.session_state:
        st.session_state.opening_style_index = 0
    if "personal_info" not in st.session_state:
        st.session_state.personal_info = {
            "full_name": "",
            "age": 45,
            "gender": "Prefer not to say",
            "pregnancy_status": "Unknown/Not provided",
            "country": "",
            "state_province_county": "",
            "city_town_village": "",
            "state_province": "",
            "city": "",
            "county": "",
            "locality": "",
            "street_address": "",
            "postal_code": "",
            "address_raw": "",
            "address": "",
            "phone": "",
            "allergies": "",
            "current_medications": "",
            "village": "Turin, Piedmont (Italy)",
            "health_notes": "",
        }
    if "text_input_rev" not in st.session_state:
        st.session_state.text_input_rev = 0
    if "file_input_rev" not in st.session_state:
        st.session_state.file_input_rev = 0
    if "voice_input_rev" not in st.session_state:
        st.session_state.voice_input_rev = 0
    if "landing_complete" not in st.session_state:
        st.session_state.landing_complete = False
    if "app_section" not in st.session_state:
        st.session_state.app_section = "conversation"

    # Backward compatibility for sessions created before structured address fields existed.
    profile = st.session_state.personal_info
    profile.setdefault("country", "")
    profile.setdefault("state_province_county", "")
    profile.setdefault("city_town_village", "")
    profile.setdefault("state_province", "")
    profile.setdefault("city", "")
    profile.setdefault("county", "")
    profile.setdefault("locality", "")
    profile.setdefault("street_address", "")
    profile.setdefault("postal_code", "")
    profile.setdefault("address_raw", "")
    profile.setdefault("address", "")
    profile.setdefault("pregnancy_status", "Unknown/Not provided")
    profile.setdefault("allergies", "")
    profile.setdefault("current_medications", "")
    sync_merged_location_fields(profile)
    st.session_state.personal_info = profile


def handle_user_turn(
    user_text: str,
    village: Dict[str, Any],
    resources: List[Dict[str, Any]],
    kb_sections: List[Dict[str, Any]],
    villages: List[Dict[str, Any]],
    medical_chunks: List[Dict[str, Any]],
) -> None:
    profile = st.session_state.personal_info
    auto_update_profile_from_message(user_text, profile, villages)
    st.session_state.personal_info = profile
    previous_triage = dict(st.session_state.last_triage or {})
    previous_assistant_text = ""
    for m in reversed(st.session_state.chat_history):
        if m.get("role") == "assistant" and not m.get("kind"):
            previous_assistant_text = str(m.get("content", "")).strip()
            break
    opening_style_index = int(st.session_state.opening_style_index or 0)
    previous_context = dict(st.session_state.clinical_context or empty_clinical_context())
    clinical_context = update_clinical_context(previous_context, user_text, profile)
    last_question_target = str(st.session_state.last_question_target or "")
    clinical_context["slot_status"] = update_slot_status(previous_context, clinical_context, last_question_target, user_text)
    st.session_state.clinical_context = clinical_context
    profile_gaps = get_relevant_profile_gaps(user_text, profile)
    assistant_msg = ""
    map_snapshot: Optional[Dict[str, Any]] = None
    should_run = is_clinical_followup_turn(user_text, clinical_context, last_question_target)
    followup_turn = bool(should_run) and bool(
        previous_context.get("symptoms") or previous_context.get("duration_text") or previous_context.get("severity_text")
    )
    clinical_delta: Dict[str, Any] = {}

    should_run_clinical = should_run
    response_mode = detect_response_mode(
        user_text=user_text,
        location_intent=is_location_support_intent(user_text),
        medicine_intent=is_medicine_question(user_text),
        triage=None,
        clinical_context=clinical_context,
    )

    if should_run_clinical:
        warmup_triage = pulse_run("Running Triage Tool...", triage_tool, user_text, kb_sections, clinical_context)
        warmup_triage = apply_risk_floor(warmup_triage, clinical_context)
        clinical_delta = compute_clinical_delta(previous_context, clinical_context, previous_triage, warmup_triage)
        should_geo = should_invoke_geo_rag(user_text, warmup_triage)
        agent_result = run_agent_loop(
            user_text,
            village,
            resources,
            kb_sections,
            profile,
            invoke_geo=should_geo,
            triage_override=warmup_triage,
        )
        st.session_state.last_triage = agent_result["triage"]
        st.session_state.last_locator = agent_result["locator"]
        st.session_state.last_geo_rag_basis = agent_result.get("geo_rag_basis", "")
        st.session_state.last_geo_rag_ref = agent_result.get("geo_rag_ref", "")
        assistant_msg = generate_natural_chat_reply(
            user_text=user_text,
            profile=profile,
            triage=agent_result["triage"],
            locator=agent_result["locator"],
            profile_gaps=profile_gaps,
            medical_chunks=medical_chunks,
            clinical_context=clinical_context,
            last_question_target=last_question_target,
            clinical_delta=clinical_delta,
            followup_turn=followup_turn,
            previous_assistant_text=previous_assistant_text,
            opening_style_index=opening_style_index,
        )
        response_mode = detect_response_mode(
            user_text=user_text,
            location_intent=is_location_support_intent(user_text),
            medicine_intent=is_medicine_question(user_text),
            triage=agent_result["triage"],
            clinical_context=clinical_context,
        )
        assistant_msg = continuity_guardrail_reply(assistant_msg, clinical_context, response_mode)
        if (
            should_geo
            and agent_result["locator"]
            and
            not bool(agent_result["locator"].get("no_country_coverage"))
            and should_attach_geo_map(user_text, agent_result["triage"], agent_result["locator"], assistant_msg)
        ):
            map_snapshot = build_geo_map_snapshot(village, agent_result["locator"], resources)
        if not should_geo:
            st.session_state.last_locator = {}
            st.session_state.last_geo_rag_basis = ""
            st.session_state.last_geo_rag_ref = ""
    else:
        clinical_delta = compute_clinical_delta(previous_context, clinical_context, previous_triage, None)
        assistant_msg = generate_natural_chat_reply(
            user_text=user_text,
            profile=profile,
            profile_gaps=profile_gaps,
            medical_chunks=medical_chunks,
            clinical_context=clinical_context,
            last_question_target=last_question_target,
            clinical_delta=clinical_delta,
            followup_turn=followup_turn,
            previous_assistant_text=previous_assistant_text,
            opening_style_index=opening_style_index,
        )
        assistant_msg = continuity_guardrail_reply(assistant_msg, clinical_context, response_mode)
    assistant_msg = clean_response_grammar(assistant_msg)
    st.session_state.opening_style_index = (opening_style_index + 1) % 10000

    if response_mode in {"clinical_assessment", "medication_guidance", "cause_explainer"}:
        gaps = infer_intake_gaps(user_text, profile, clinical_context=clinical_context)
        if (
            last_question_target
            and (last_question_target in gaps)
            and not target_filled_between(last_question_target, previous_context, clinical_context)
        ):
            st.session_state.last_question_target = last_question_target
        else:
            st.session_state.last_question_target = choose_next_question_target(gaps, last_question_target)
        new_target = str(st.session_state.last_question_target or "")
        slot_status = dict(st.session_state.clinical_context.get("slot_status", {}) or {})
        if new_target and not slot_has_value(new_target, st.session_state.clinical_context):
            slot_status[new_target] = "asked"
        st.session_state.clinical_context["slot_status"] = slot_status
    else:
        st.session_state.last_question_target = ""

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    if map_snapshot:
        st.session_state.chat_history.append(
            {"role": "assistant", "kind": "geo_map", "snapshot": map_snapshot}
        )


def render_landing_page() -> None:
    st.markdown('<div class="landing-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="landing-title">Hello, I&apos;m RuralLink. Tell me who you are and how you feel recently? I&apos;m listening.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="landing-intro"><div class="landing-cards">', unsafe_allow_html=True)
    st.markdown(
        '<div class="landing-info-card landing-reveal landing-delay-1"><p class="landing-paragraph">RuralLink Agent is your autonomous healthcare coordinator designed specifically for the 137 million citizens living in rural Europe. We bridge the gap between underserved residents and essential medical care by transforming natural conversation into clear, actionable paths to health.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="landing-info-card landing-reveal landing-delay-2"><div class="landing-section-title">Core Capabilities</div><p class="landing-bullet"><strong>Empathetic AI Triage:</strong> We translate your plain-language symptoms into professional clinical guidance, removing the barriers of medical literacy.</p><p class="landing-bullet"><strong>Intelligent Healthcare Navigation:</strong> We identify the nearest open clinics, pharmacies, or mobile health units based on your real-world location and transportation barriers.</p><p class="landing-bullet"><strong>Structured Care Briefings:</strong> We generate professional health summaries that you can share with pharmacists or doctors to ensure seamless care handoffs.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="landing-info-card landing-reveal landing-delay-3"><div class="landing-section-title">Grounded in Truth: Our Triple-Pillar Knowledge Base</div><p class="landing-paragraph">RuralLink is not a generic chatbot. Every response is verified against three authoritative knowledge pillars to ensure safety and compliance:</p><p class="landing-bullet"><strong>Medical Knowledge Base:</strong> Grounded in clinical guidelines from the European Medicines Agency (EMA) and the World Health Organization (WHO). This ensures all triage advice and over-the-counter information is evidence-based.</p><p class="landing-bullet"><strong>Policy &amp; Legal Knowledge Base:</strong> Fully aligned with the EU AI Act and national health regulations (e.g., Italy&apos;s Decreto Legislativo n. 502). We prioritize transparency, privacy (GDPR), and human oversight in every interaction.</p><p class="landing-bullet"><strong>Geographic Resource Database:</strong> Integrated with regional health resource maps to provide real-time updates on facility locations, contact details, and current availability.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown('<div class="landing-actions"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.6, 1.2], gap="small")
    with c2:
        if st.button("Have a try now!", key="landing_enter_home", use_container_width=True):
            st.session_state.landing_complete = True
            st.session_state.app_section = "conversation"
            st.rerun()
    with c3:
        if st.button("Complete Personal Information", key="landing_personal_info", use_container_width=True):
            st.session_state.landing_complete = True
            st.session_state.app_section = "personal_info"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def build_recent_condition_tags(clinical_context: Dict[str, Any], triage: Optional[Dict[str, Any]] = None) -> List[str]:
    ctx = clinical_context or {}
    tags: List[str] = []
    symptoms = [str(x).strip() for x in (ctx.get("symptoms", []) or []) if str(x).strip()]
    for s in symptoms[:4]:
        tags.append(f"Symptom: {s}")

    duration = str(ctx.get("duration_text", "")).strip()
    if duration:
        tags.append(f"Duration: {duration}")

    temp_c = float(ctx.get("temperature_c", 0.0) or 0.0)
    if temp_c > 0:
        tags.append(f"Temperature: {temp_c:.1f}C")

    risk = risk_label_from_triage(triage or {})
    if risk and risk != "lower risk":
        tags.append(f"System risk: {risk.title()}")

    red_flags = [str(x).strip() for x in (ctx.get("red_flags", []) or []) if str(x).strip()]
    for r in red_flags[:3]:
        tags.append(f"Warning sign: {r}")

    for cc in [str(x).strip() for x in (ctx.get("chronic_conditions", []) or []) if str(x).strip()][:4]:
        tags.append(f"Chronic condition: {cc}")

    for cp in [str(x).strip() for x in (ctx.get("complications", []) or []) if str(x).strip()][:4]:
        tags.append(f"Complication watch: {cp}")

    if str(ctx.get("allergies_text", "")).strip():
        tags.append(f"Allergy: {str(ctx.get('allergies_text', '')).strip()[:48]}")
    if str(ctx.get("medications_text", "")).strip():
        tags.append(f"Medication: {str(ctx.get('medications_text', '')).strip()[:48]}")

    deduped: List[str] = []
    seen = set()
    for t in tags:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(t)
    return deduped[:14]


def render_recent_condition_sidebar(clinical_context: Dict[str, Any], triage: Optional[Dict[str, Any]] = None) -> None:
    tags = build_recent_condition_tags(clinical_context, triage)
    st.markdown('<div class="recent-condition-title">Recent Condition</div>', unsafe_allow_html=True)
    if not tags:
        st.caption("No recent condition tags yet.")
        return
    chips = "".join([f'<span class="recent-tag">{html.escape(tag)}</span>' for tag in tags])
    st.markdown(f'<div class="recent-condition-wrap">{chips}</div>', unsafe_allow_html=True)


def render_sidebar_navigation() -> None:
    with st.sidebar:
        if st.button("Personal Information", key="nav_personal_information", use_container_width=True):
            st.session_state.app_section = "personal_info"
            st.rerun()
        if st.button("Health Record", key="nav_health_record", use_container_width=True):
            st.session_state.app_section = "health_record"
            st.rerun()
        st.markdown("---")
        render_recent_condition_sidebar(
            st.session_state.get("clinical_context", {}) or {},
            st.session_state.get("last_triage", {}) or {},
        )


def render_profile_editor(profile: Dict[str, Any], villages: List[Dict[str, Any]], key_prefix: str) -> Dict[str, Any]:
    profile["full_name"] = st.text_input("Full Name", value=profile.get("full_name", ""), key=f"{key_prefix}_full_name")
    profile["age"] = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=int(profile.get("age", 45) or 45),
        key=f"{key_prefix}_age",
    )
    genders = ["Female", "Male", "Non-binary", "Prefer not to say"]
    current_gender = profile.get("gender", "Prefer not to say")
    gender_index = genders.index(current_gender) if current_gender in genders else 3
    profile["gender"] = st.selectbox("Gender", genders, index=gender_index, key=f"{key_prefix}_gender")
    preg_options = ["Unknown/Not provided", "Not pregnant", "Pregnant/Breastfeeding", "Not applicable"]
    preg_current = profile.get("pregnancy_status", "Unknown/Not provided")
    preg_index = preg_options.index(preg_current) if preg_current in preg_options else 0
    profile["pregnancy_status"] = st.selectbox(
        "Pregnancy Status",
        preg_options,
        index=preg_index,
        key=f"{key_prefix}_pregnancy_status",
    )

    st.caption("Real Address (used by assistant for personalized location guidance)")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        profile["country"] = st.text_input("Country", value=profile.get("country", ""), key=f"{key_prefix}_country")
        profile["state_province_county"] = st.text_input(
            "State/Province/County",
            value=profile.get("state_province_county", ""),
            key=f"{key_prefix}_state_province_county",
        )
        profile["city_town_village"] = st.text_input(
            "City/Town/Village",
            value=profile.get("city_town_village", ""),
            key=f"{key_prefix}_city_town_village",
        )
    with c2:
        profile["street_address"] = st.text_input(
            "Street Address",
            value=profile.get("street_address", ""),
            key=f"{key_prefix}_street_address",
        )
        profile["postal_code"] = st.text_input("Postal Code", value=profile.get("postal_code", ""), key=f"{key_prefix}_postal_code")
        profile["phone"] = st.text_input("Phone", value=profile.get("phone", ""), key=f"{key_prefix}_phone")
    profile["allergies"] = st.text_input(
        "Allergies",
        value=profile.get("allergies", ""),
        key=f"{key_prefix}_allergies",
    )
    profile["current_medications"] = st.text_input(
        "Current Medications",
        value=profile.get("current_medications", ""),
        key=f"{key_prefix}_current_medications",
    )

    sync_merged_location_fields(profile)
    profile["address"] = compose_full_address(profile)
    if profile["address"]:
        st.caption(f"Address Preview: {profile['address']}")
    if profile.get("address_raw", "").strip():
        st.caption(f"Detected from conversation: {profile['address_raw']}")
    profile["health_notes"] = st.text_area(
        "Health Notes (allergies, chronic conditions)",
        value=profile.get("health_notes", ""),
        height=95,
        key=f"{key_prefix}_health_notes",
    )
    inferred_area = infer_base_area_from_profile(profile, villages)
    profile["village"] = inferred_area.get("name", "Turin, Piedmont (Italy)")
    return profile


def render_personal_information_page(villages: List[Dict[str, Any]]) -> None:
    st.markdown("## Personal Information")
    st.caption("Edit and browse your profile. Chat can auto-fill this information when relevant.")
    profile = dict(st.session_state.personal_info)
    profile = render_profile_editor(profile, villages, key_prefix="personal_info_page")
    st.session_state.personal_info = profile
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button("Save Changes", key="personal_info_save", use_container_width=True):
            st.session_state.personal_info = profile
            st.success("Personal information updated.")
    with c2:
        if st.button("Back to Home", key="personal_info_back", use_container_width=True):
            st.session_state.app_section = "conversation"
            st.rerun()


def render_health_record_page(kb_sections: List[Dict[str, Any]]) -> None:
    st.markdown("## Health Record")
    st.write(f"Saved documents: **{len(st.session_state.document_summaries)}**")
    if st.session_state.last_triage:
        st.write(
            f"Latest triage: **{st.session_state.last_triage.get('severity', 'unknown')}** "
            f"(confidence {st.session_state.last_triage.get('confidence_score', 0.0)})"
        )
    if st.session_state.last_locator:
        st.write(
            f"Latest nearest resource: **{st.session_state.last_locator.get('nearest_resource', {}).get('name', 'N/A')}**"
        )

    if st.button("Generate/Refresh Care Briefing", key="health_generate_brief", use_container_width=True):
        case_context = build_case_context()
        result = pulse_run("Generating briefing...", briefing_tool, case_context, kb_sections)
        st.session_state.last_briefing = result["briefing_markdown"]
        if result.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
            st.warning("Briefing confidence is low; keep a human reviewer in the loop.")

    if st.session_state.last_briefing:
        with st.expander("View latest briefing", expanded=False):
            st.markdown(st.session_state.last_briefing)
        st.download_button(
            label="Download Briefing (.md)",
            data=st.session_state.last_briefing,
            file_name="rurallink_case_briefing.md",
            mime="text/markdown",
            key="health_download_briefing",
        )

    if st.session_state.document_summaries:
        rows = []
        for item in st.session_state.document_summaries[-30:]:
            analysis = item.get("analysis", {})
            rows.append(
                {
                    "filename": item.get("filename", ""),
                    "doc_type": analysis.get("doc_type", "Unknown"),
                    "confidence": analysis.get("confidence_score", ""),
                    "time": item.get("timestamp", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No document records yet.")

    if st.session_state.dispatch_status:
        st.success(st.session_state.dispatch_status)

    if st.button("Back to Home", key="health_back", use_container_width=True):
        st.session_state.app_section = "conversation"
        st.rerun()


def render_conversation_workspace(
    villages: List[Dict[str, Any]],
    resources: List[Dict[str, Any]],
    kb_sections: List[Dict[str, Any]],
    medical_chunks: List[Dict[str, Any]],
) -> None:
    profile = dict(st.session_state.personal_info)
    sync_merged_location_fields(profile)
    inferred_area = infer_base_area_from_profile(profile, villages)
    profile["village"] = inferred_area.get("name", "Turin, Piedmont (Italy)")
    st.session_state.personal_info = profile

    village = resolve_village(villages, profile.get("village", "Turin, Piedmont (Italy)"))
    st.session_state.selected_village = village

    for m in st.session_state.chat_history:
        if m.get("kind") == "geo_map":
            snapshot = m.get("snapshot") or {}
            render_geo_rag_panel_snapshot(snapshot)
            continue
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        role = "assistant" if m["role"] == "assistant" else "user"
        with st.chat_message(role, avatar=role):
            st.markdown(content)

    with st.container(key="input_console"):
        st.markdown("**Input Console**")
        left_col, mid_col, right_col = st.columns(3, gap="small")
        file_key = f"agent_file_input_{st.session_state.file_input_rev}"
        text_key = f"agent_text_input_{st.session_state.text_input_rev}"
        voice_key = f"agent_voice_input_{st.session_state.voice_input_rev}"

        with left_col:
            st.caption("Upload Files")
            upload_files = st.file_uploader(
                "Upload Files",
                type=["jpg", "jpeg", "png", "pdf"],
                accept_multiple_files=True,
                key=file_key,
                label_visibility="collapsed",
            )
            send_files = st.button("Send Files", key="send_files_btn", use_container_width=True)

        with mid_col:
            st.caption("Type Message")
            text_value = st.text_area(
                "Type Message",
                key=text_key,
                placeholder="Example:\nI am Maria, 68.\nI feel unwell and I have no car.",
                label_visibility="collapsed",
                height=96,
            )
            send_text = st.button("Send Text", key="send_text_btn", use_container_width=True)

        with right_col:
            st.caption("Record Voice")
            voice_blob = st.audio_input(
                "Record Voice",
                sample_rate=16000,
                key=voice_key,
                label_visibility="collapsed",
            )
            send_voice = st.button("Send Voice", key="send_voice_btn", use_container_width=True)

    if send_text:
        text_msg = (text_value or "").strip()
        if not text_msg:
            st.warning("Please type a message first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": text_msg})
            handle_user_turn(text_msg, village, resources, kb_sections, villages, medical_chunks)
            st.session_state.text_input_rev += 1
            st.rerun()

    if send_files:
        if not upload_files:
            st.warning("Please select at least one file first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": f"[Uploaded {len(upload_files)} file(s)]"})
            files_count = process_uploaded_files_in_chat(upload_files, kb_sections)
            if files_count > 0:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I processed your uploaded documents. "
                            "Share your symptoms or care goal in one sentence, and I will run triage + Geo-RAG next."
                        ),
                    }
                )
            st.session_state.file_input_rev += 1
            st.rerun()

    if send_voice:
        if voice_blob is None:
            st.warning("Please record a voice message first.")
        else:
            transcript_text = pulse_run("Transcribing voice input...", transcribe_audio_file, voice_blob)
            if transcript_text:
                st.session_state.chat_history.append({"role": "user", "content": f"Voice transcript: {transcript_text}"})
                handle_user_turn(transcript_text, village, resources, kb_sections, villages, medical_chunks)
            else:
                st.session_state.chat_history.append({"role": "user", "content": "[Voice message received]"})
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I received your voice message, but I could not transcribe it this time. "
                            "Please try again or type your message."
                        ),
                    }
                )
            st.session_state.voice_input_rev += 1
            st.rerun()

    triage = st.session_state.last_triage
    if triage.get("needs_urgent_care"):
        st.warning("Urgent pathway detected. External dispatch needs your explicit confirmation.")
        if st.button("I agree to transmit case to local emergency dispatch", key="confirm_dispatch"):
            status_msg = mock_ehds_transmit()
            st.session_state.dispatch_status = status_msg
            st.session_state.chat_history.append({"role": "assistant", "content": f"I completed the requested transmission. {status_msg}"})
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="RuralLink Agent", layout="wide")
    apply_ui_theme()
    init_state()

    villages = load_json(VILLAGES_PATH, [])
    resources = load_json(RESOURCES_PATH, [])
    kb_sections = load_json(KB_PATH, {}).get("sections", [])
    medical_chunks = load_medical_chunks()

    if not st.session_state.landing_complete:
        render_landing_page()
        return

    st.title("RuralLink Agent")
    st.caption("EU Rural Healthcare Autonomy Prototype | Conversation-first Agent with integrated Geo-RAG")

    render_sidebar_navigation()

    section = st.session_state.app_section
    if section == "personal_info":
        render_personal_information_page(villages)
        return
    if section == "health_record":
        render_health_record_page(kb_sections)
        return

    render_conversation_workspace(villages, resources, kb_sections, medical_chunks)
    st.caption("OpenAI key source: OPENAI_API_KEY")
    st.caption(
        "Prototype disclaimer: This system is for educational simulation. It does not replace licensed clinical judgment. "
        "External actions are mock operations with human confirmation controls."
    )


if __name__ == "__main__":
    main()
