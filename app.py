import base64
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
  background: #f5f7fc;
}
.card {
  background: #ffffff;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(16, 38, 84, 0.08);
  padding: 1rem 1.1rem;
  border: 1px solid #e4eaf7;
  margin-bottom: 1rem;
}
.agent-zone {
  background: linear-gradient(130deg, #f6f1ff 0%, #ecf9f1 100%);
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(16, 38, 84, 0.08);
  padding: 1rem 1.1rem;
  border: 1px solid #e4eaf7;
  margin-bottom: 1rem;
}
.stButton > button {
  background: #0b2e6f;
  color: #fff;
  border-radius: 10px;
  border: 1px solid #0b2e6f;
  font-weight: 600;
}
.stButton > button:hover {
  background: #1d458f;
  border-color: #1d458f;
}
.pulse {
  display: inline-block;
  padding: 0.45rem 0.8rem;
  border-radius: 999px;
  border: 1px solid #dae4ff;
  color: #1d3668;
  background: #edf2ff;
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0% { transform: scale(1); opacity: 0.75; }
  50% { transform: scale(1.04); opacity: 1; }
  100% { transform: scale(1); opacity: 0.75; }
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


def geo_locator_tool(
    village: Dict[str, Any],
    resources: List[Dict[str, Any]],
    triage: Dict[str, Any],
    user_text: str,
) -> Dict[str, Any]:
    wants_medicine = any(k in user_text.lower() for k in ["medicine", "pharmacy", "drug", "flu"])
    transport_barrier = any(k in user_text.lower() for k in ["no car", "can't travel", "cannot travel", "30 km", "no transport"])

    preferred_types: List[str]
    if triage.get("needs_urgent_care"):
        preferred_types = ["hospital", "mobile_clinic"]
    elif wants_medicine:
        preferred_types = ["pharmacy", "clinic"]
    else:
        preferred_types = [triage.get("recommended_care_type", "clinic"), "clinic", "pharmacy"]

    if transport_barrier and "mobile_clinic" not in preferred_types:
        preferred_types.insert(0, "mobile_clinic")
    if transport_barrier and "transport" not in preferred_types:
        preferred_types.append("transport")

    lat, lon = float(village["lat"]), float(village["lon"])
    ranked: List[Dict[str, Any]] = []

    for row in resources:
        rlat, rlon = float(row["location"][0]), float(row["location"][1])
        d = haversine_km(lat, lon, rlat, rlon)
        row_copy = dict(row)
        row_copy["distance_km"] = round(d, 2)
        ranked.append(row_copy)

    ranked.sort(key=lambda x: x["distance_km"])

    filtered = [x for x in ranked if x.get("type") in preferred_types]
    chosen = filtered[0] if filtered else ranked[0]

    legal_ref = "RURAL_CARE_GUIDE_2026_2" if transport_barrier else "RURAL_CARE_GUIDE_2026_1"
    legal_basis = (
        "When transportation barriers are present, coordination should prioritize mobile clinics and community transport."
        if transport_barrier
        else "Urgency and symptom profile should guide referral to the closest appropriate care resource."
    )

    return {
        "preferred_types": preferred_types,
        "transport_barrier": transport_barrier,
        "nearest_resource": chosen,
        "top_candidates": ranked[:3],
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


def run_agent_loop(user_text: str, village: Dict[str, Any], resources: List[Dict[str, Any]], kb_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    triage = pulse_run("Running Triage Tool...", triage_tool, user_text, kb_sections)
    locator = pulse_run("Running Geo-Locator Tool...", geo_locator_tool, village, resources, triage, user_text)

    plan = [
        f"Step 1: Triage Tool -> severity '{triage.get('severity', 'unknown')}'.",
        f"Step 2: Geo-Locator Tool -> nearest resource '{locator.get('nearest_resource', {}).get('name', 'N/A')}'.",
        "Step 3: Briefing Tool -> ready to generate cross-session handoff brief.",
    ]

    nearest = locator.get("nearest_resource", {})
    response = (
        f"{triage.get('warm_reply', 'I am here with you.')}\n\n"
        f"I assessed your symptoms as **{triage.get('severity', 'unknown')}** priority. "
        f"The closest matching resource is **{nearest.get('name', 'N/A')}** ({nearest.get('distance_km', 'N/A')} km away). "
        f"Contact: {nearest.get('phone', 'N/A')}.\n\n"
        f"I can also prepare a pharmacist/clinician briefing for you right now."
    )

    legal_block = (
        f"Legal Basis: {triage.get('legal_basis', '')} ({triage.get('legal_ref', '')}); "
        f"{locator.get('legal_basis', '')} ({locator.get('legal_ref', '')})"
    )

    if triage.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
        response += "\n\nI recommend a human healthcare worker review this recommendation before any external action."

    return {
        "assistant_reply": response,
        "plan_steps": plan,
        "triage": triage,
        "locator": locator,
        "legal_block": legal_block,
    }


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello, I am RuralLink Agent. Tell me what you are feeling and I will coordinate triage, nearby resources, and a care briefing for you.",
            }
        ]
    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = []
    if "last_triage" not in st.session_state:
        st.session_state.last_triage = {}
    if "last_locator" not in st.session_state:
        st.session_state.last_locator = {}
    if "last_briefing" not in st.session_state:
        st.session_state.last_briefing = ""
    if "dispatch_status" not in st.session_state:
        st.session_state.dispatch_status = ""


def main() -> None:
    st.set_page_config(page_title="RuralLink Agent", layout="wide")
    apply_ui_theme()
    init_state()

    villages = load_json(VILLAGES_PATH, [])
    resources = load_json(RESOURCES_PATH, [])
    kb_sections = load_json(KB_PATH, {}).get("sections", [])

    st.title("RuralLink Agent")
    st.caption("EU Rural Healthcare Autonomy Prototype | Agentic Planning + Tool Use + Human-in-the-loop")

    with st.sidebar:
        st.markdown("### Patient Context")
        village_names = [v["name"] for v in villages] or ["Village A"]
        selected_name = st.selectbox("Select your village", village_names, index=0)
        village = next((v for v in villages if v.get("name") == selected_name), {"name": selected_name, "lat": 45.123, "lon": 7.456})
        st.session_state.selected_village = village

        st.markdown("### Configuration")
        st.write("OpenAI key source: `OPENAI_API_KEY` environment variable")
        if get_openai_api_key():
            st.success("OpenAI API key detected.")
        else:
            st.warning("OpenAI API key not set. App will run in fallback mode.")

        st.markdown("### EU AI Act Control")
        st.info("Any external transmission requires explicit user confirmation (HITL).")

    tab_console, tab_map, tab_brief = st.tabs(["Agent Console", "Resource Map", "Case Briefing"])

    with tab_console:
        st.markdown('<div class="agent-zone">', unsafe_allow_html=True)
        st.markdown("**Live Agent Conversation**")
        st.write(f"Village context: **{village['name']}** ({village['lat']}, {village['lon']})")

        uploaded = st.file_uploader(
            "Upload supporting file (image/pdf)",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            key="intake_uploads",
        )
        if uploaded and st.button("Analyze Uploaded Files", key="analyze_uploads"):
            for f in uploaded:
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
                    f"Key facts: {', '.join(result.get('key_facts', []))}\n"
                    f"Legal Basis: {result.get('legal_basis')} ({result.get('legal_ref')})"
                )
                if result.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
                    msg += "\nI recommend human review due to low confidence or limited extraction."
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.rerun()

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_text = st.chat_input("Example: I feel unwell and I have no car to reach town.")
        if user_text:
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            agent_result = run_agent_loop(user_text, village, resources, kb_sections)

            st.session_state.last_triage = agent_result["triage"]
            st.session_state.last_locator = agent_result["locator"]

            plan_text = "\n".join(f"- {x}" for x in agent_result["plan_steps"])
            assistant_msg = (
                f"{agent_result['assistant_reply']}\n\n"
                f"Planned Tool Calls:\n{plan_text}\n\n"
                f"{agent_result['legal_block']}"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
            st.rerun()

        triage = st.session_state.last_triage
        if triage.get("needs_urgent_care"):
            st.warning("Urgent pathway detected. External dispatch requires your explicit consent.")
            if st.button("I agree to transmit case to local emergency dispatch", key="confirm_dispatch"):
                status_msg = mock_ehds_transmit()
                st.session_state.dispatch_status = status_msg
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I completed the requested transmission. {status_msg}\n\nLegal Basis: EHDS secure transmission principle (EHDS_PRINCIPLE_INTEROP)",
                })
                st.rerun()

        if st.session_state.dispatch_status:
            st.success(st.session_state.dispatch_status)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_map:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Geo-Locator View (Mocked GPS + Resource Mapping)**")

        rows = [{"label": f"{village['name']} (Patient)", "lat": village["lat"], "lon": village["lon"]}]
        for r in resources:
            rows.append({"label": r["name"], "lat": r["location"][0], "lon": r["location"][1]})
        map_df = pd.DataFrame(rows)
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]], zoom=10)

        locator = st.session_state.last_locator
        if locator.get("top_candidates"):
            st.markdown("**Nearest options from latest agent run**")
            st.dataframe(pd.DataFrame(locator["top_candidates"])[["name", "type", "status", "phone", "distance_km"]], use_container_width=True)
        else:
            st.info("Run one conversation in Agent Console to see ranked nearby resources.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_brief:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Briefing Generator (Cross-session handoff)**")
        st.caption("Creates a structured note for pharmacist/clinician handoff.")

        if st.button("Generate Professional Briefing", key="generate_brief"):
            case_context = build_case_context()
            result = pulse_run("Generating briefing...", briefing_tool, case_context, kb_sections)
            st.session_state.last_briefing = result["briefing_markdown"]
            if result.get("fallback_action") == "FLAG_FOR_HUMAN_REVIEW":
                st.warning("Briefing confidence is below threshold. Please request human review before operational use.")
            st.caption(f"Legal Basis: {result.get('legal_basis')} ({result.get('legal_ref')})")

        if st.session_state.last_briefing:
            st.markdown(st.session_state.last_briefing)
            st.download_button(
                label="Download Briefing (.md)",
                data=st.session_state.last_briefing,
                file_name="rurallink_case_briefing.md",
                mime="text/markdown",
            )
        else:
            st.info("No briefing yet. Click 'Generate Professional Briefing' after at least one conversation.")

        if st.session_state.document_summaries:
            st.markdown("**Recent document analyses**")
            doc_rows = []
            for item in st.session_state.document_summaries[-5:]:
                analysis = item.get("analysis", {})
                doc_rows.append(
                    {
                        "filename": item.get("filename"),
                        "doc_type": analysis.get("doc_type"),
                        "confidence": analysis.get("confidence_score"),
                        "fallback_action": analysis.get("fallback_action"),
                    }
                )
            st.table(pd.DataFrame(doc_rows))

        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Prototype disclaimer: This system is for educational simulation. It does not replace licensed clinical judgment. "
        "External actions are mock operations with human confirmation controls."
    )


if __name__ == "__main__":
    main()
