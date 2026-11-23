from typing import List
from unittest.mock import patch

import pytest

from rebuff.detect_pi_openai import parse_score_from_completion
from rebuff.sdk import RebuffSdk


@pytest.fixture()
def rebuff() -> RebuffSdk:
    return RebuffSdk("openai-api-key", "pinecone-api-key", "pinecone-index")


# Complete, valid scores: single values and decimals (optionally padded with
# whitespace) must keep working.
VALID_COMPLETIONS = {
    "1": 1.0,
    "0": 0.0,
    "0.95": 0.95,
    "0.5": 0.5,
    " 0.9 ": 0.9,
    "0.85\n": 0.85,
}

# Invalid responses: multiple numbers, explanations before/after the number,
# and empty content must be rejected.
INVALID_COMPLETIONS = [
    "",
    "   ",
    "\n",
    "0.9 0.1",
    "0.9\n0.1",
    "Score: 0.8",
    "0.8 because the input is suspicious",
    "The score is 0.8",
    "not a number",
]


def test_parse_score_from_completion_accepts_complete_scores():
    for completion, expected in VALID_COMPLETIONS.items():
        assert parse_score_from_completion(completion) == expected


def test_parse_score_from_completion_rejects_invalid_scores():
    for completion in INVALID_COMPLETIONS:
        with pytest.raises(ValueError):
            parse_score_from_completion(completion)

    with pytest.raises(ValueError):
        parse_score_from_completion(None)


def test_detect_injection_with_interleaved_model_responses(rebuff: RebuffSdk):
    max_model_score = 0.9
    # Interleave valid and invalid model responses and check the final state
    # of each detection.
    interleaved_completions: List[str] = [
        "0.95",
        "0.9 0.1",
        "Score: 0.8",
        "1",
        "",
        "0.0",
        "0.5 because reasons",
        "0.85",
    ]
    expected_scores = {
        "0.95": 0.95,
        "1": 1.0,
        "0.0": 0.0,
        "0.85": 0.85,
    }

    def fake_call_openai_to_detect_pi(prompt, model, api_key):
        return {"completion": interleaved_completions.pop(0)}

    with patch(
        "rebuff.sdk.call_openai_to_detect_pi",
        side_effect=fake_call_openai_to_detect_pi,
    ):
        for completion in list(interleaved_completions):
            if completion in expected_scores:
                response = rebuff.detect_injection(
                    "user input",
                    max_model_score=max_model_score,
                    check_heuristic=False,
                    check_vector=False,
                    check_llm=True,
                )
                expected_score = expected_scores[completion]
                assert response.openai_score == expected_score
                assert response.injection_detected == (
                    expected_score > max_model_score
                )
            else:
                with pytest.raises(ValueError):
                    rebuff.detect_injection(
                        "user input",
                        max_model_score=max_model_score,
                        check_heuristic=False,
                        check_vector=False,
                        check_llm=True,
                    )
