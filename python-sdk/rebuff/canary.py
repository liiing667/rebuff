"""Helpers for generating, embedding and detecting canary words.

A canary is identified by the exact text that is embedded in the prompt (the
``marker``) as well as by the random word used to build that marker.  Keeping
those two values together avoids the classic failure modes where a generated
word is truncated, where an empty word matches every completion, or where a
short word is found as a substring of an unrelated word.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Callable, Optional, cast


DEFAULT_CANARY_FORMAT = "<!-- {canary_word} -->"
CANARY_WORD_PLACEHOLDER = "{canary_word}"
MIN_CANARY_LENGTH = 1

_WORD_RE = re.compile(r"\w", re.ASCII)
_MARKER_CHARS = frozenset("<>{}[]()\"'`")


@dataclass(frozen=True)
class PreparedCanary:
    """The identity of a canary that was (or is about to be) embedded."""

    canary_word: str
    marker: str


class CanaryWord(str):
    """A canary word that remembers the exact marker it was embedded as.

    It behaves like a normal string (and is serialised as one), but carrying
    the marker lets a later completion check use the same identity even for a
    custom ``canary_format`` that has no delimiter around the word.
    """

    marker: str

    def __new__(cls, word: str, marker: str = "") -> "CanaryWord":
        instance = cast("CanaryWord", super().__new__(cls, word))
        instance.marker = marker
        return instance


def generate_canary_word(length: int = 8) -> str:
    """Generate a hexadecimal canary word of exactly ``length`` characters.

    ``secrets.token_hex`` takes a *byte* count and always returns twice as many
    hex characters.  Splitting an odd requested length with integer division
    used to silently drop a character; generating the extra byte and slicing
    keeps the requested length.  A length below one is rejected instead of
    producing an empty canary that matches every completion.
    """

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < MIN_CANARY_LENGTH:
        raise ValueError(f"length must be at least {MIN_CANARY_LENGTH}")

    return secrets.token_hex((length + 1) // 2)[:length]


def format_canary_marker(
    canary_word: str, canary_format: str = DEFAULT_CANARY_FORMAT
) -> str:
    """Render ``canary_format`` with ``canary_word``.

    ``str.replace`` is used instead of ``str.format`` so that an otherwise
    ordinary prompt template containing other braces is not rejected.
    """

    if not isinstance(canary_word, str):
        raise TypeError("canary_word must be a string")
    if not isinstance(canary_format, str):
        raise TypeError("canary_format must be a string")
    return canary_format.replace(CANARY_WORD_PLACEHOLDER, canary_word)


def extract_canary_word(
    prompt: str, canary_format: str = DEFAULT_CANARY_FORMAT
) -> Optional[str]:
    """Return an explicitly embedded canary word from a legacy prompt.

    Prompts that already carry a concrete marker (for example
    ``<!-- 1a2b3c4d -->``) keep working and reuse that marker's identity rather
    than gaining a second, unrelated canary.  A bare ``{canary_word}``
    placeholder is not an identity and is handled separately by
    :func:`embed_canary`.
    """

    if not isinstance(prompt, str) or not isinstance(canary_format, str):
        return None
    if CANARY_WORD_PLACEHOLDER in prompt:
        return None
    if CANARY_WORD_PLACEHOLDER not in canary_format:
        return None

    before, after = canary_format.split(CANARY_WORD_PLACEHOLDER, 1)
    if not before and not after:
        # A bare word is indistinguishable from arbitrary prompt text.
        return None
    pattern = re.escape(before) + r"(?P<canary_word>.+?)" + re.escape(after)
    match = re.search(pattern, prompt, flags=re.DOTALL)
    if match is None:
        return None

    canary_word = match.group("canary_word").strip()
    if not canary_word or any(character.isspace() for character in canary_word):
        return None
    return canary_word


def _looks_like_marker(value: str) -> bool:
    """Whether ``value`` is a complete marker rather than a bare word."""

    return any(character.isspace() or character in _MARKER_CHARS for character in value)


def _is_word_character(character: str) -> bool:
    return bool(character) and _WORD_RE.match(character) is not None


def _contains_standalone_token(text: str, token: str) -> bool:
    """Find ``token`` without matching it as part of a longer word.

    For example the canary ``dead`` must not match the ordinary word
    ``deadline`` and the canary ``1a2b`` must not match inside a longer hex
    digest.  Punctuation, whitespace and string boundaries are accepted.
    """

    if not token:
        return False

    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return False

        before = text[index - 1] if index > 0 else ""
        after_index = index + len(token)
        after = text[after_index] if after_index < len(text) else ""
        if not _is_word_character(before) and not _is_word_character(after):
            return True
        start = index + 1


def is_canary_leaked(
    completion: str,
    canary_word: str,
    canary_format: Optional[str] = None,
) -> bool:
    """Check a completion against the same identity that was embedded.

    ``canary_word`` may be the bare generated word or an already formatted
    marker.  When ``canary_format`` is supplied the rendered marker is checked
    exactly as well.  An empty or missing canary never counts as a leak.
    """

    if not completion or not canary_word or not isinstance(canary_word, str):
        return False
    if not isinstance(completion, str):
        return False

    attached_marker = getattr(canary_word, "marker", None)
    if attached_marker and attached_marker in completion:
        return True

    if canary_format is not None:
        marker = format_canary_marker(canary_word, canary_format)
        if marker and marker != canary_word and marker in completion:
            return True

    if _looks_like_marker(canary_word):
        return canary_word in completion
    return _contains_standalone_token(completion, canary_word)


def embed_canary(
    prompt: str,
    canary_word: Optional[str] = None,
    canary_format: str = DEFAULT_CANARY_FORMAT,
    word_generator: Optional[Callable[[], str]] = None,
) -> tuple[str, PreparedCanary]:
    """Insert or resolve a canary while preserving legacy prompt templates.

    * An existing concrete marker is reused so the returned identity always
      belongs to the marker already in the prompt.
    * A legacy ``{canary_word}`` placeholder is filled in place.
    * Otherwise the marker is prepended exactly once.

    When ``canary_format`` contains no placeholder it is itself an explicit
    marker; that marker becomes the canary identity so it can be checked and
    logged consistently.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(canary_format, str):
        raise TypeError("canary_format must be a string")
    if canary_word is not None and (
        not isinstance(canary_word, str) or not canary_word
    ):
        raise ValueError("canary_word must be a non-empty string")

    has_placeholder = CANARY_WORD_PLACEHOLDER in canary_format
    if canary_word is not None:
        # Accept an already formatted explicit marker as well as a bare word.
        explicit_word = extract_canary_word(canary_word, canary_format)
        if explicit_word is not None:
            canary_word = explicit_word
    existing = extract_canary_word(prompt, canary_format)

    if canary_word is None:
        if existing is not None:
            canary_word = existing
        elif not has_placeholder and canary_format:
            canary_word = canary_format
        else:
            canary_word = (word_generator or generate_canary_word)()
    if canary_word is None:
        raise ValueError("could not resolve a canary word")

    if has_placeholder:
        marker = format_canary_marker(canary_word, canary_format)
        identity = canary_word
    else:
        marker = canary_format or canary_word
        identity = marker

    if CANARY_WORD_PLACEHOLDER in prompt:
        prompt_with_canary = prompt.replace(CANARY_WORD_PLACEHOLDER, canary_word)
    elif existing is not None and existing != canary_word and has_placeholder:
        old_marker = format_canary_marker(existing, canary_format)
        prompt_with_canary = prompt.replace(old_marker, marker)
    elif marker and marker in prompt:
        prompt_with_canary = prompt
    elif marker:
        prompt_with_canary = marker + "\n" + prompt
    else:
        prompt_with_canary = prompt

    return prompt_with_canary, PreparedCanary(CanaryWord(identity, marker), marker)
