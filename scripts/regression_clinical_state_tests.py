import importlib.util
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("rurallink_app", APP_PATH)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_context_persists_symptom_and_duration():
    profile = {
        "age": 68,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "",
        "allergies": "",
        "current_medications": "",
    }
    ctx = app.empty_clinical_context()
    ctx = app.update_clinical_context(ctx, "headache", profile)
    ctx = app.update_clinical_context(ctx, "two weeks", profile)
    assert_true("headache" in ctx.get("symptoms", []), "Symptom should persist across turns.")
    assert_true(int(ctx.get("duration_days", 0)) >= 14, "Duration should be parsed as 14+ days.")


def test_risk_floor_for_headache_two_weeks():
    ctx = {
        "symptoms": ["headache"],
        "duration_days": 14,
    }
    triage = {
        "severity": "low",
        "needs_urgent_care": False,
        "recommended_care_type": "clinic",
        "triage_summary": "Initial triage completed.",
    }
    out = app.apply_risk_floor(triage, ctx)
    assert_true(out.get("severity") in {"moderate", "high", "critical"}, "Risk floor should raise low to at least moderate.")


def test_followup_turn_detection():
    ctx = app.empty_clinical_context()
    ctx["symptoms"] = ["headache"]
    result = app.is_clinical_followup_turn("two weeks", ctx, "symptom duration")
    assert_true(result, "Short follow-up answer should stay in clinical flow.")


def test_non_answer_not_treated_as_followup():
    ctx = app.empty_clinical_context()
    ctx["symptoms"] = ["headache"]
    result = app.is_clinical_followup_turn("hi", ctx, "symptom duration")
    assert_true(not result, "Irrelevant short text should not be treated as duration answer.")


def test_response_mode_from_followup_context():
    ctx = app.empty_clinical_context()
    ctx["symptoms"] = ["headache"]
    mode = app.detect_response_mode("two weeks", False, False, {"severity": "low"}, clinical_context=ctx)
    assert_true(mode == "clinical_assessment", "Follow-up with existing symptom should remain clinical_assessment.")


def test_opening_has_at_least_ten_variants():
    ctx = {
        "symptoms": ["fever"],
        "duration_text": "2 days",
        "severity_text": "moderate",
    }
    seen = set()
    for i in range(20):
        seen.add(app.known_clinical_facts_line(ctx, i))
    assert_true(len(seen) >= 10, f"Expected >=10 opening variants, got {len(seen)}.")


def test_slot_state_machine_transitions():
    profile = {
        "age": 68,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "",
        "allergies": "",
        "current_medications": "",
    }
    prev = app.empty_clinical_context()
    curr = app.update_clinical_context(prev, "fever", profile)
    slots = app.update_slot_status(prev, curr, "symptom duration", "fever")
    assert_true(slots.get("main symptom") in {"filled", "confirmed"}, "Main symptom should be filled after symptom mention.")
    assert_true(slots.get("symptom duration") == "asked", "Asked slot should be tracked as asked.")

    prev2 = dict(curr)
    curr2 = app.update_clinical_context(curr, "2 days", profile)
    slots2 = app.update_slot_status(prev2, curr2, "symptom duration", "2 days")
    assert_true(slots2.get("symptom duration") in {"filled", "confirmed"}, "Duration should become filled after answer.")


def test_fever_followup_prefers_temperature_not_self_scored_severity():
    profile = {
        "age": 68,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "",
        "allergies": "",
        "current_medications": "",
        "gender": "Prefer not to say",
        "address_raw": "",
        "address": "",
        "country": "",
    }
    ctx = app.empty_clinical_context()
    ctx = app.update_clinical_context(ctx, "fever", profile)
    gaps = app.infer_intake_gaps("fever", profile, clinical_context=ctx)
    target = app.choose_next_question_target(gaps, "")
    q = app.format_gap_question(gaps, max_items=1, last_target="")
    assert_true(target == "symptom duration", "First follow-up for fever should ask duration first.")
    assert_true("severity level" not in q.lower(), "Prompt should not ask user to self-score severity.")


def test_recent_condition_tags_include_duration_and_chronic_flags():
    profile = {
        "age": 68,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "hypertension and diabetes",
        "allergies": "",
        "current_medications": "",
    }
    ctx = app.empty_clinical_context()
    ctx = app.update_clinical_context(ctx, "fever for 2 days", profile)
    tags = app.build_recent_condition_tags(ctx, {"severity": "moderate", "needs_urgent_care": False})
    joined = " | ".join(tags).lower()
    assert_true("duration: 2 days" in joined, "Recent condition tags should include duration.")
    assert_true("chronic condition: hypertension" in joined or "chronic condition: diabetes" in joined, "Recent tags should include chronic condition extraction.")


def test_continuity_guardrail_removes_symptom_reask():
    ctx = app.empty_clinical_context()
    ctx["symptoms"] = ["headache"]
    bad_reply = "Could you tell me what symptoms you have been experiencing?"
    fixed = app.continuity_guardrail_reply(bad_reply, ctx, "clinical_assessment")
    assert_true("what symptoms" not in fixed.lower(), "Guardrail should remove repeated symptom question.")


def test_geo_not_default_for_medicine_question():
    triage = {"severity": "low", "needs_urgent_care": False}
    invoke = app.should_invoke_geo_rag("what medicine should I take for fever?", triage)
    assert_true(not invoke, "Geo-RAG should not trigger by default for general medicine question.")


def test_acceptance_gate_no_symptom_reask_after_known_symptom():
    profile = {
        "age": 70,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "",
        "allergies": "",
        "current_medications": "",
        "gender": "Prefer not to say",
        "address_raw": "",
        "address": "",
        "country": "",
    }
    ctx = app.empty_clinical_context()
    messages = ["headache", "two weeks", "moderate"]
    repeats = 0
    last_target = ""
    prev_ctx = app.empty_clinical_context()
    prev_triage = {}
    prev_reply = ""
    for m in messages:
        prev_ctx = dict(ctx)
        ctx = app.update_clinical_context(ctx, m, profile)
        triage = app.apply_risk_floor(app.heuristic_triage(m, []), ctx)
        delta = app.compute_clinical_delta(prev_ctx, ctx, prev_triage, triage)
        followup_turn = bool(prev_ctx.get("symptoms") or prev_ctx.get("duration_text"))
        reply = app.generate_natural_chat_reply(
            user_text=m,
            profile=profile,
            triage=triage,
            locator={},
            profile_gaps=[],
            medical_chunks=[],
            clinical_context=ctx,
            last_question_target=last_target,
            clinical_delta=delta,
            followup_turn=followup_turn,
            previous_assistant_text=prev_reply,
        )
        reply = app.continuity_guardrail_reply(reply, ctx, "clinical_assessment")
        if "what symptoms" in reply.lower():
            repeats += 1
        gaps = app.infer_intake_gaps(m, profile, clinical_context=ctx)
        last_target = app.choose_next_question_target(gaps, last_target)
        prev_triage = triage
        prev_reply = reply
    rate = repeats / max(len(messages), 1)
    assert_true(rate < 0.05, f"Symptom re-ask rate should be <5%, got {rate:.2%}.")


def test_followup_reply_is_incremental_not_full_template():
    profile = {
        "age": 70,
        "pregnancy_status": "Unknown/Not provided",
        "health_notes": "",
        "allergies": "",
        "current_medications": "",
        "gender": "Prefer not to say",
        "address_raw": "",
        "address": "",
        "country": "",
    }
    prev = app.empty_clinical_context()
    prev = app.update_clinical_context(prev, "fever", profile)
    prev_triage = app.apply_risk_floor(app.heuristic_triage("fever", []), prev)
    curr = app.update_clinical_context(prev, "2 days", profile)
    triage = app.apply_risk_floor(app.heuristic_triage("2 days", []), curr)
    delta = app.compute_clinical_delta(prev, curr, prev_triage, triage)
    reply = app.generate_natural_chat_reply(
        user_text="2 days",
        profile=profile,
        triage=triage,
        locator={},
        profile_gaps=[],
        medical_chunks=[],
        clinical_context=curr,
        last_question_target="symptom duration",
        clinical_delta=delta,
        followup_turn=True,
        previous_assistant_text="Risk level: Moderate Risk",
    )
    assert_true("**What you can do now**" not in reply, "Follow-up reply should not reuse full template section.")
    assert_true("2 days" in reply.lower(), "Follow-up reply should explicitly acknowledge new fact.")
    assert_true("your whether" not in reply.lower(), "Follow-up reply should avoid grammar artifact 'your whether'.")


def test_clean_response_preserves_basic_newlines():
    raw = (
        "Intro sentence.\nLine two with  extra   spaces.\n\nLine three."
    )
    cleaned = app.clean_response_grammar(raw)
    assert_true("\n" in cleaned, "Cleaner should preserve normal line breaks.")
    assert_true("extra spaces" in cleaned, "Cleaner should normalize repeated spaces.")


def run_all():
    tests = [
        test_context_persists_symptom_and_duration,
        test_risk_floor_for_headache_two_weeks,
        test_followup_turn_detection,
        test_non_answer_not_treated_as_followup,
        test_response_mode_from_followup_context,
        test_opening_has_at_least_ten_variants,
        test_slot_state_machine_transitions,
        test_fever_followup_prefers_temperature_not_self_scored_severity,
        test_recent_condition_tags_include_duration_and_chronic_flags,
        test_continuity_guardrail_removes_symptom_reask,
        test_geo_not_default_for_medicine_question,
        test_acceptance_gate_no_symptom_reask_after_known_symptom,
        test_followup_reply_is_incremental_not_full_template,
        test_clean_response_preserves_basic_newlines,
    ]
    for t in tests:
        t()
    print(f"PASS: {len(tests)} regression checks")


if __name__ == "__main__":
    run_all()
