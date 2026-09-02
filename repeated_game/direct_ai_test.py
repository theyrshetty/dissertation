"""
Unit tests for the deterministic parts of direct_ai.py: prompt construction
and answer extraction. These do NOT call the network — they test extract_action()
against hand-written mock LLM outputs, including messy/edge cases.
"""

from direct_ai import build_prompt, extract_action

def test_build_prompt_includes_paragraph():
    para = "Some unique paragraph text 12345."
    prompt = build_prompt(para)
    assert para in prompt
    assert "FINAL ANSWER:" in prompt
    print("PASS: build_prompt includes the paragraph and instruction line")


def test_extract_clean_final_answer():
    text = "Some reasoning here about incentives.\n\nFINAL ANSWER: COOPERATE"
    assert extract_action(text) == "cooperate"
    print("PASS: clean FINAL ANSWER line extracted")


def test_extract_final_answer_lowercase_and_extra_space():
    text = "blah blah\nfinal answer:   defect  "
    assert extract_action(text) == "defect"
    print("PASS: case-insensitive / whitespace-tolerant match")


def test_extract_missing_final_answer_falls_back_to_last_mention():
    text = (
        "I think cooperation is risky here. On balance, given the payoffs, "
        "the safer choice is to defect."
    )
    assert extract_action(text) == "defect"
    print("PASS: fallback picks the last standalone mention when no FINAL ANSWER line")


def test_extract_no_mention_returns_none():
    text = "I cannot make a recommendation without more information."
    assert extract_action(text) is None
    print("PASS: returns None (parse failure) when neither word appears")


def test_extract_prefers_final_answer_over_earlier_contradicting_mentions():
    # Model reasons through both options in prose, but the explicit
    # FINAL ANSWER line should win over any earlier fallback scan.
    text = (
        "One could argue for cooperate here, but ultimately defect is safer "
        "given the short horizon.\n\nFINAL ANSWER: COOPERATE"
    )
    assert extract_action(text) == "cooperate"
    print("PASS: explicit FINAL ANSWER line takes priority over prose mentions")


if __name__ == "__main__":
    test_build_prompt_includes_paragraph()
    test_extract_clean_final_answer()
    test_extract_final_answer_lowercase_and_extra_space()
    test_extract_missing_final_answer_falls_back_to_last_mention()
    test_extract_no_mention_returns_none()
    test_extract_prefers_final_answer_over_earlier_contradicting_mentions()
    print("\nAll direct_ai unit tests passed.")