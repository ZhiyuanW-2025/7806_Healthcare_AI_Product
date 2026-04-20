# RuralLink Agent (EU Rural Healthcare Autonomy Prototype)

RuralLink Agent is an **agentic** Streamlit prototype that demonstrates:
- planning + tool-use
- natural-language triage dialog
- multimodal document intake (image-first)
- lightweight local RAG against a policy knowledge base
- human-in-the-loop (HITL) control before external actions

## 1) Project Structure

```text
rurallink-agent/
  app.py
  requirements.txt
  .gitignore
  data/
    villages.json
    mock_resources.json
    eu_health_kb.json
```

## 2) Agent Tools

### Tool A: Triage Tool
- Input: user symptom message
- Output: severity, urgent flag, legal basis, confidence score
- Includes fallback: if confidence < 0.85, `FLAG_FOR_HUMAN_REVIEW`

### Tool B: Geo-Locator Tool
- Input: selected village coordinates + triage result
- Output: nearest matched resource and top candidates
- Uses `mock_resources.json` and distance ranking

### Tool C: Briefing Generator
- Input: cross-session context (chat + triage + location + documents)
- Output: professional handoff briefing markdown
- Includes legal citations and confidence fallback

## 3) Mocking Strategy

### GPS & Map Mocking
- User selects a village in sidebar.
- Backend maps village name to fixed coordinates from `villages.json`.
- Map is centered/visualized using those coordinates in Streamlit.

### Resource DB Mocking
- `mock_resources.json` contains sample pharmacy/hospital/mobile clinic/transport entries.
- Geo-Locator ranks nearest candidates by distance.

### External API Mocking
- Dispatch transmission is simulated with a 2-second delay.
- UI message: `Connecting to EHDS data interface...`
- Success output: `Status 200: Case transmitted to local emergency dispatch.`

## 4) Run Locally

```bash
cd /Users/wuzhiyuan/.codex/workspaces/default/rurallink-agent
pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
streamlit run app.py --server.port 8510 --server.headless true
```

Open: `http://localhost:8510`

If `OPENAI_API_KEY` is not set, the app still runs in fallback mode.

## 5) HITL Safety Control (EU AI Act Alignment)

The app **never transmits** external case data automatically.
Before dispatch simulation, user must click:
`I agree to transmit case to local emergency dispatch`

This demonstrates human oversight and risk control in high-impact operations.
