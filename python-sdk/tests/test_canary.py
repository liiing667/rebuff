import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import pytest

from rebuff.canary import (
    generate_canary_word,
    is_canary_leaked,
)
from rebuff.rebuff import Rebuff


@pytest.fixture()
def rebuff() -> Rebuff:
    return Rebuff(api_token="test-token")


@pytest.mark.parametrize("length", list(range(1, 33)))
def test_generated_canary_has_exact_length(length: int) -> None:
    canary_word = generate_canary_word(length)

    assert len(canary_word) == length
    assert re.fullmatch(r"[0-9a-f]+", canary_word)


@pytest.mark.parametrize("length", [0, -1])
def test_empty_canary_length_is_rejected(length: int) -> None:
    with pytest.raises(ValueError):
        generate_canary_word(length)


@pytest.mark.parametrize("length", [1.5, "8", True])
def test_non_integer_canary_length_is_rejected(length: object) -> None:
    with pytest.raises(TypeError):
        generate_canary_word(length)  # type: ignore[arg-type]


def test_partial_substring_is_not_a_leak() -> None:
    assert not is_canary_leaked("the deadline is near", "dead")
    assert not is_canary_leaked("prefixdead123", "dead")
    assert not is_canary_leaked("deadline", "deadline2")

    assert is_canary_leaked("the token is dead.", "dead")
    assert is_canary_leaked("<!-- dead -->", "dead")


def test_explicit_and_legacy_markers_are_reused(rebuff: Rebuff) -> None:
    prompt, canary_word = rebuff.add_canary_word(
        "<!-- {canary_word} -->\nlegacy prompt", "explicit"
    )
    assert prompt == "<!-- explicit -->\nlegacy prompt"
    assert canary_word == "explicit"

    prompt, canary_word = rebuff.add_canary_word(
        "<!-- existing -->\nlegacy prompt"
    )
    assert prompt == "<!-- existing -->\nlegacy prompt"
    assert canary_word == "existing"


def test_leak_log_uses_same_canary_identity(
    rebuff: Rebuff, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: List[Tuple[str, str, str]] = []

    def fake_log(user_input: str, completion: str, canary_word: str) -> None:
        logged.append((user_input, completion, canary_word))

    monkeypatch.setattr(rebuff, "log_leakage", fake_log)

    assert rebuff.is_canary_word_leaked("input", "value abc123 here", "abc123")
    assert logged == [("input", "value abc123 here", "abc123")]

    assert not rebuff.is_canary_word_leaked(
        "input", "value abc1234 here", "abc123"
    )
    assert len(logged) == 1


def test_custom_format_keeps_marker_identity(rebuff: Rebuff) -> None:
    prompt, canary_word = rebuff.add_canary_word(
        "hello", canary_format="pre{canary_word}post"
    )

    assert prompt == f"pre{canary_word}post\nhello"
    assert rebuff.is_canary_word_leaked(
        "input", f"pre{canary_word}post", canary_word, False
    )


def test_concurrent_insertion_does_not_cross_markers(rebuff: Rebuff) -> None:
    prompt_template: Any = pytest.importorskip(
        "langchain_core.prompts"
    ).PromptTemplate(
        input_variables=["user_input"],
        template="Tell me a joke about {user_input}",
    )
    original_template = prompt_template.template

    def add(canary_word: str) -> Tuple[Any, str]:
        return rebuff.add_canary_word(prompt_template, canary_word)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(add, [f"canary{i:03d}" for i in range(64)]))

    for prompt, canary_word in results:
        assert prompt.template.count(canary_word) == 1
        other_words = [other for _, other in results if other != canary_word]
        assert all(other not in prompt.template for other in other_words)

    # The shared template is never mutated by a call.
    assert prompt_template.template == original_template
