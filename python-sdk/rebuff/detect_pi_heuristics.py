import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

# Unified input budget (in characters) shared by input acceptance, fragment
# generation, scoring, and the detection response. Inputs longer than this are
# truncated to the budget instead of being rejected, so legitimate long
# documents remain usable while pathological inputs can no longer stall the
# heuristic check. Empty or blank inputs are rejected so they cannot bypass
# the budget.
MAX_INPUT_LENGTH = 10_000


def enforce_input_budget(
    user_input: str, max_length: int = MAX_INPUT_LENGTH
) -> Tuple[str, bool]:
    """
    Validate user input and bound it to the shared input budget.

    Args:
        user_input (str): The raw user input to validate.
        max_length (int): The input budget in characters. Defaults to MAX_INPUT_LENGTH.

    Returns:
        Tuple[str, bool]: The budgeted input (truncated to max_length when needed)
            and a flag telling whether truncation happened.

    Raises:
        ValueError: If the input is empty/blank or the budget is not positive.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    if max_length < 1:
        raise ValueError("max_length must be a positive integer")

    if len(user_input) > max_length:
        return user_input[:max_length], True

    return user_input, False


def generate_injection_keywords() -> List[str]:
    """
    Generate all possible combinations of sentences based on a list of predefined verbs, adjectives, prepositions, and objects that can be used for prompt injection.

    Args:
        None

    Returns:
        List of sentences
    """
    verbs = [
        "Ignore",
        "Disregard",
        "Skip",
        "Forget",
        "Neglect",
        "Overlook",
        "Omit",
        "Bypass",
        "Pay no attention to",
        "Do not follow",
        "Do not obey",
    ]

    adjectives = [
        "",
        "prior",
        "previous",
        "preceding",
        "above",
        "foregoing",
        "earlier",
        "initial",
    ]

    prepositions = [
        "",
        "and start over",
        "and start anew",
        "and begin afresh",
        "and start from scratch",
    ]

    objects = [
        "content",
        "text",
        "instructions",
        "instruction",
        "directives",
        "directive",
        "commands",
        "command",
        "context",
        "conversation",
        "input",
        "inputs",
        "data",
        "message",
        "messages",
        "communication",
        "response",
        "responses",
        "request",
        "requests",
    ]

    # Generate all possible combinations of sentences
    injection_keywords = []
    for verb in verbs:
        for adjective in adjectives:
            for object in objects:
                for preposition in prepositions:
                    all_words = (
                        verb + " " + adjective + " " + object + " " + preposition
                    )
                    injection_keywords.append(all_words)

    return injection_keywords


def normalize_string(input_string: str) -> str:
    """
    Normalized input string by converting to lower case, remove characters that are not letters, remove excession white space etc.

    Args:
        input_string (str): String to be normalized

    Returns:
        normalized_string (str)
    """

    # Convert to lowercase
    result = input_string.lower()

    # Remove characters that are not letters, digits, spaces, or underscores
    result = re.sub(r"[^\w\s]|_", "", result)

    # Replace multiple consecutive spaces with a single space
    result = re.sub(r"\s+", " ", result)

    # Trim leading and trailing spaces
    normalized_string = result.strip()

    return normalized_string


def get_input_substrings(
    normalized_input: str,
    keyword_length: int,
    max_length: int = MAX_INPUT_LENGTH,
) -> List[str]:
    """
    Iterate over the input string and get substrings which have same length as as the keywords string

    Args:
        normalized_input (str): Normalized input string
        keyword_length (int): The number of words in the injection string
        max_length (int): The shared input budget in characters. The input is
            truncated to this budget before fragments are generated.

    Returns:
        List of input substrings that have the same length as the number of keywords in injection string
    """
    # Bound fragment generation to the same budget used everywhere else. All
    # state is local to this call, so a retried call never continues with
    # fragments left over from a previous, failed attempt.
    budgeted_input = normalized_input[:max_length]
    words_in_input_string = budgeted_input.split(" ")
    input_substrings = []
    number_of_substrings = len(words_in_input_string) - keyword_length + 1
    for i in range(number_of_substrings):
        input_substrings.append(" ".join(words_in_input_string[i : i + keyword_length]))

    return input_substrings


def get_matched_words_score(
    substring: str, keyword_parts: List[str], max_matched_words: int
) -> float:
    matched_words_count = len(
        [part for part, word in zip(keyword_parts, substring.split()) if word == part]
    )

    if matched_words_count > 0:
        base_score = 0.5 + 0.5 * min(matched_words_count / max_matched_words, 1)
    else:
        base_score = 0

    return base_score


def _build_normalized_injection_keywords() -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Precompute the normalized injection keywords and their word parts once at
    module load, instead of regenerating them on every scoring call.
    """
    normalized_keywords = []
    for keyword_string in generate_injection_keywords():
        normalized_keyword_string = normalize_string(keyword_string)
        normalized_keywords.append(
            (normalized_keyword_string, tuple(normalized_keyword_string.split(" ")))
        )
    return normalized_keywords


_NORMALIZED_INJECTION_KEYWORDS = _build_normalized_injection_keywords()


def detect_prompt_injection_using_heuristic_on_input(
    input: str, max_length: int = MAX_INPUT_LENGTH
) -> float:
    # Input acceptance: reject empty/blank input and truncate overlong input
    # to the shared budget, so huge inputs cannot stall the heuristic check.
    budgeted_input, _ = enforce_input_budget(input, max_length)

    highest_score = 0.0
    max_matched_words = 5

    normalized_input_string = normalize_string(budgeted_input)
    input_words = normalized_input_string.split(" ")
    number_of_words = len(input_words)

    # Call-local similarity cache: repeated windows (common in padded or
    # repetitive inputs) are only scored once, and a retried call always
    # starts from a fresh cache instead of continuing with stale fragments.
    similarity_cache: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = {}

    for normalized_keyword_string, keywords in _NORMALIZED_INJECTION_KEYWORDS:
        keyword_length = len(keywords)
        if keyword_length > number_of_words:
            continue

        # Generate substrings of similar length (to keyword length) in the input string
        for i in range(number_of_words - keyword_length + 1):
            window = tuple(input_words[i : i + keyword_length])

            # Count positionally matched words between the keyword and the window
            matched_words_count = 0
            for part, word in zip(keywords, window):
                if part == word:
                    matched_words_count += 1

            if matched_words_count == 0:
                # The adjusted score would be <= 0 here, so it can never raise
                # the highest score. Skipping the expensive similarity call
                # keeps long documents fast without changing the result.
                continue

            cache_key = (keywords, window)
            similarity_score = similarity_cache.get(cache_key)
            if similarity_score is None:
                similarity_score = SequenceMatcher(
                    None, " ".join(window), normalized_keyword_string
                ).ratio()
                similarity_cache[cache_key] = similarity_score

            base_score = 0.5 + 0.5 * min(matched_words_count / max_matched_words, 1)

            # Adjust the score using the similarity score
            adjusted_score = base_score - similarity_score * (
                1 / (max_matched_words * 2)
            )

            if adjusted_score > highest_score:
                highest_score = adjusted_score

    return highest_score
