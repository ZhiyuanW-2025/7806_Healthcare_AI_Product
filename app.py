import base64
import io
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_PATH = BASE_DIR / "data" / "mock_resources.json"
VILLAGES_PATH = BASE_DIR / "data" / "villages.json"
KB_PATH = BASE_DIR / "data" / "eu_health_kb.json"
LOW_CONFIDENCE_THRESHOLD = 0.85

SYSTEM_PROMPT = """
You are RuralLink Agent, the chief rural healthcare coordinator for EU community settings.
Tone: professional, calm, and empathetic.
Rules:
- Always provide legal_basis and legal_ref for every decision.
- If confidence_score < 0.85, set fallback_action=FLAG_FOR_HUMAN_REVIEW and explicitly advise human review.
- Never claim a real dispatch happened unless user explicitly confirms and the system executes mock dispatch.
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


def triage_tool(user_text: str, kb_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    legal_hits = retrieve_kb_context(user_text, kb_sections, top_k=2)
    fallback = heuristic_triage(user_text, legal_hits)
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
{user_text}

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
        return {
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
    except Exception:
        return fallback


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
        pv = expand_location_aliases(profile.get(field, ""))
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
) -> Dict[str, Any]:
    triage = pulse_run("Running Triage Tool...", triage_tool, user_text, kb_sections)
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
        f"The closest matching resource is **{nearest.get('name', 'N/A')}** ({nearest.get('distance_km', 'N/A')} km away). "
        f"Contact: {nearest.get('phone', 'N/A')}.\n\n"
        f"I can also prepare a pharmacist/clinician briefing for you right now."
    )

    if triage.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
        response += "\n\nI recommend a human healthcare worker review this recommendation before any external action."

    return {
        "assistant_reply": response,
        "triage": triage,
        "locator": locator,
        "geo_rag_basis": geo_rag_basis,
        "geo_rag_ref": geo_rag_ref,
    }


def compose_full_address(profile: Dict[str, Any]) -> str:
    ordered_parts = [
        str(profile.get("street_address", "")).strip(),
        str(profile.get("locality", "")).strip(),
        str(profile.get("county", "")).strip(),
        str(profile.get("city", "")).strip(),
        str(profile.get("state_province", "")).strip(),
        str(profile.get("postal_code", "")).strip(),
        str(profile.get("country", "")).strip(),
    ]
    clean_parts = [p for p in ordered_parts if p]
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
            pv = normalize_text(profile.get(f, ""))
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


def auto_update_profile_from_message(user_text: str, profile: Dict[str, Any], villages: List[Dict[str, Any]]) -> bool:
    candidate = extract_address_candidate_from_message(user_text)
    if not candidate:
        return False

    changed = False
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
    location_keywords = [
        "where", "near", "nearest", "closest", "nearby", "pharmacy", "hospital", "clinic",
        "drugstore", "medicine", "route", "address", "travel", "transport", "how far",
    ]
    return any(k in t for k in location_keywords)


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


def should_attach_geo_map(
    user_text: str,
    triage: Optional[Dict[str, Any]],
    locator: Optional[Dict[str, Any]],
    assistant_msg: str,
) -> bool:
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
            "address": user_address,
            "phone": profile.get("phone", ""),
            "village": profile.get("village", ""),
            "health_notes": profile.get("health_notes", ""),
        },
        "triage": triage or {},
        "locator": locator or {},
        "intent_flags": {
            "location_intent": location_intent,
            "age_sensitive_intent": is_age_sensitive_intent(user_text),
            "gender_sensitive_intent": is_gender_sensitive_intent(user_text),
        },
        "relevant_missing_profile_fields": profile_gaps,
    }
    prompt = f"""
User message:
{user_text}

Context:
{json.dumps(context, ensure_ascii=False)}

Write a natural conversational reply (2-5 sentences), warm and practical.
Important style rules:
- No section headers, no labels, no template-like structure.
- Use personal info (address, age, gender) only when relevant to this message.
- For location questions, use the exact user address from context when available as location context.
- The resource address may be a mock address from locator.nearest_resource.mock_address.
- If relevant info is missing in relevant_missing_profile_fields, ask for it in one friendly question.
- Do NOT always mention severity level or nearest resource.
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
    if "personal_info" not in st.session_state:
        st.session_state.personal_info = {
            "full_name": "",
            "age": 45,
            "gender": "Prefer not to say",
            "country": "",
            "state_province": "",
            "city": "",
            "county": "",
            "locality": "",
            "street_address": "",
            "postal_code": "",
            "address_raw": "",
            "address": "",
            "phone": "",
            "village": "Turin, Piedmont (Italy)",
            "health_notes": "",
        }
    if "text_input_rev" not in st.session_state:
        st.session_state.text_input_rev = 0
    if "file_input_rev" not in st.session_state:
        st.session_state.file_input_rev = 0
    if "voice_input_rev" not in st.session_state:
        st.session_state.voice_input_rev = 0

    # Backward compatibility for sessions created before structured address fields existed.
    profile = st.session_state.personal_info
    profile.setdefault("country", "")
    profile.setdefault("state_province", "")
    profile.setdefault("city", "")
    profile.setdefault("county", "")
    profile.setdefault("locality", "")
    profile.setdefault("street_address", "")
    profile.setdefault("postal_code", "")
    profile.setdefault("address_raw", "")
    profile.setdefault("address", "")
    st.session_state.personal_info = profile


def handle_user_turn(
    user_text: str,
    village: Dict[str, Any],
    resources: List[Dict[str, Any]],
    kb_sections: List[Dict[str, Any]],
    villages: List[Dict[str, Any]],
) -> None:
    profile = st.session_state.personal_info
    auto_update_profile_from_message(user_text, profile, villages)
    st.session_state.personal_info = profile
    profile_gaps = get_relevant_profile_gaps(user_text, profile)
    assistant_msg = ""
    map_snapshot: Optional[Dict[str, Any]] = None

    if is_healthcare_or_navigation_intent(user_text):
        agent_result = run_agent_loop(user_text, village, resources, kb_sections, profile)
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
        )
        if (
            not bool(agent_result["locator"].get("no_country_coverage"))
            and should_attach_geo_map(user_text, agent_result["triage"], agent_result["locator"], assistant_msg)
        ):
            map_snapshot = build_geo_map_snapshot(village, agent_result["locator"], resources)
    else:
        assistant_msg = generate_natural_chat_reply(
            user_text=user_text,
            profile=profile,
            profile_gaps=profile_gaps,
        )
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    if map_snapshot:
        st.session_state.chat_history.append(
            {"role": "assistant", "kind": "geo_map", "snapshot": map_snapshot}
        )


def main() -> None:
    st.set_page_config(page_title="RuralLink Agent", layout="wide")
    apply_ui_theme()
    init_state()

    villages = load_json(VILLAGES_PATH, [])
    resources = load_json(RESOURCES_PATH, [])
    kb_sections = load_json(KB_PATH, {}).get("sections", [])

    st.title("RuralLink Agent")
    st.caption("EU Rural Healthcare Autonomy Prototype | Conversation-first Agent with integrated Geo-RAG")
    st.caption(
        "How to use: Tell me in plain language how you feel, when it started, and anything that makes it better or worse. "
        "You can also share practical barriers like transportation issues, and upload a file or voice message if helpful. "
        "I will reply like a supportive doctor: first listen carefully, then give clear next steps, and only bring in triage or nearby-resource options when truly needed."
    )

    with st.sidebar:
        st.markdown("### Personal Information")
        profile = st.session_state.personal_info
        profile["full_name"] = st.text_input("Full Name", value=profile.get("full_name", ""))
        profile["age"] = st.number_input("Age", min_value=0, max_value=120, value=int(profile.get("age", 45)))
        profile["gender"] = st.selectbox(
            "Gender",
            ["Female", "Male", "Non-binary", "Prefer not to say"],
            index=["Female", "Male", "Non-binary", "Prefer not to say"].index(
                profile.get("gender", "Prefer not to say")
            )
            if profile.get("gender", "Prefer not to say") in ["Female", "Male", "Non-binary", "Prefer not to say"]
            else 3,
        )
        st.caption("Real Address (used by assistant for personalized location guidance)")
        addr_col1, addr_col2 = st.columns(2, gap="small")
        with addr_col1:
            profile["country"] = st.text_input("Country", value=profile.get("country", ""))
            profile["state_province"] = st.text_input("State/Province", value=profile.get("state_province", ""))
            profile["city"] = st.text_input("City", value=profile.get("city", ""))
            profile["county"] = st.text_input("County/District", value=profile.get("county", ""))
        with addr_col2:
            profile["locality"] = st.text_input("Town/Village", value=profile.get("locality", ""))
            profile["street_address"] = st.text_input("Street Address", value=profile.get("street_address", ""))
            profile["postal_code"] = st.text_input("Postal Code", value=profile.get("postal_code", ""))
        profile["address"] = compose_full_address(profile)
        if profile["address"]:
            st.caption(f"Address Preview: {profile['address']}")
        profile["phone"] = st.text_input("Phone", value=profile.get("phone", ""))
        inferred_area = infer_base_area_from_profile(profile, villages)
        profile["village"] = inferred_area.get("name", "Turin, Piedmont (Italy)")
        profile["health_notes"] = st.text_area(
            "Health Notes (allergies, chronic conditions)",
            value=profile.get("health_notes", ""),
            height=90,
        )
        st.session_state.personal_info = profile
        st.markdown("---")

        st.markdown("### Health Record")
        st.write(f"Saved documents: **{len(st.session_state.document_summaries)}**")
        if st.session_state.last_triage:
            st.write(
                f"Latest triage: **{st.session_state.last_triage.get('severity', 'unknown')}** "
                f"(confidence {st.session_state.last_triage.get('confidence_score', 0.0)})"
            )
        if st.session_state.last_locator:
            st.write(f"Latest nearest resource: **{st.session_state.last_locator.get('nearest_resource', {}).get('name', 'N/A')}**")

        if st.button("Generate/Refresh Care Briefing", key="sidebar_generate_brief"):
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
                key="sidebar_download_briefing",
            )

        if st.session_state.dispatch_status:
            st.success(st.session_state.dispatch_status)

        st.caption("OpenAI key source: OPENAI_API_KEY")

    village = resolve_village(villages, st.session_state.personal_info.get("village", "Turin, Piedmont (Italy)"))
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
                placeholder="Example:\nI feel unwell, have a fever,\nand I do not have a car.",
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
            handle_user_turn(text_msg, village, resources, kb_sections, villages)
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
                st.session_state.chat_history.append(
                    {"role": "user", "content": f"Voice transcript: {transcript_text}"}
                )
                handle_user_turn(transcript_text, village, resources, kb_sections, villages)
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
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"I completed the requested transmission. {status_msg}",
                }
            )
            st.rerun()

    st.caption(
        "Prototype disclaimer: This system is for educational simulation. It does not replace licensed clinical judgment. "
        "External actions are mock operations with human confirmation controls."
    )


if __name__ == "__main__":
    main()
