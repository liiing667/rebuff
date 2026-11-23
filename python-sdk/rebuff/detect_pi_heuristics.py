import re
import unicodedata
from typing import Dict, List


# Zero-width / bidi / invisible formatting characters that can be used to hide
# or join words without changing how a human reads the text.
INVISIBLE_CHARACTERS = (
    "\u00ad"  # soft hyphen
    "\u061c"  # arabic letter mark
    "\u115f"  # hangul choseong filler
    "\u1160"  # hangul jungseong filler
    "\u17b4"  # khmer vowel inherent aq
    "\u17b5"  # khmer vowel inherent aa
    "\u180e"  # mongolian vowel separator
    "\u200b"  # zero width space
    "\u200c"  # zero width non-joiner
    "\u200d"  # zero width joiner
    "\u200e"  # left-to-right mark
    "\u200f"  # right-to-left mark
    "\u202a"  # left-to-right embedding
    "\u202b"  # right-to-left embedding
    "\u202c"  # pop directional formatting
    "\u202d"  # left-to-right override
    "\u202e"  # right-to-left override
    "\u2060"  # word joiner
    "\u2061"  # function application
    "\u2062"  # invisible times
    "\u2063"  # invisible separator
    "\u2064"  # invisible plus
    "\u2066"  # left-to-right isolate
    "\u2067"  # right-to-left isolate
    "\u2068"  # first strong isolate
    "\u2069"  # pop directional isolate
    "\u206a"  # inhibit symmetric swapping
    "\u206b"  # activate symmetric swapping
    "\u206c"  # inhibit arabic form shaping
    "\u206d"  # activate arabic form shaping
    "\u206e"  # national digit shapes
    "\u206f"  # nominal digit shapes
    "\ufeff"  # zero width no-break space / byte order mark
)

_INVISIBLE_CHARACTERS_RE = re.compile("[" + re.escape(INVISIBLE_CHARACTERS) + "]")
# Any character that is not a letter, number or whitespace. Mirrors the JS
# regexes used by the SDK/server so both sides normalize identically.
_NON_WORD_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _dice_coefficient(first: str, second: str) -> float:
    """
    Sorensen-Dice coefficient over adjacent character bigrams.

    This is a faithful port of the ``compareTwoStrings`` function from the
    ``string-similarity`` npm package used by the JavaScript SDK and the
    server, so the local Python SDK and the API produce identical scores.
    Whitespace is ignored on both sides before comparing.

    Args:
        first (str): First string to compare.
        second (str): Second string to compare.

    Returns:
        float: Similarity score between 0 and 1.
    """
    first = re.sub(r"\s+", "", first, flags=re.UNICODE)
    second = re.sub(r"\s+", "", second, flags=re.UNICODE)

    if first == second:
        return 1
    if len(first) < 2 or len(second) < 2:
        return 0

    first_bigram_counts: Dict[str, int] = {}
    for i in range(len(first) - 1):
        bigram = first[i : i + 2]
        first_bigram_counts[bigram] = first_bigram_counts.get(bigram, 0) + 1

    intersection_size = 0
    for i in range(len(second) - 1):
        bigram = second[i : i + 2]
        count = first_bigram_counts.get(bigram, 0)
        if count > 0:
            first_bigram_counts[bigram] = count - 1
            intersection_size += 1

    return (2.0 * intersection_size) / (len(first) + len(second) - 2)


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
    Normalize an input string so visually equivalent text keeps one identity.

    The normalization is deliberately shared by the keyword fragments, the
    heuristic scorer and the final rejection decision, and it matches the
    JavaScript SDK / server implementation byte for byte:

    * Unicode compatibility folding (NFKC) turns full-width Latin letters such
      as ``Ｉｇｎｏｒｅ`` into plain ASCII and normalizes compatibility forms.
    * Zero-width, bidi and other invisible formatting characters are dropped.
    * Punctuation (including full-width punctuation such as ``，`` or
      ``。``) becomes a space instead of being deleted, so words on either
      side are never silently glued together.
    * Everything is lower-cased and runs of whitespace collapse to one space.
    * Non-English letters are preserved, so ordinary non-English text simply
      never matches the English injection keyword list.

    Args:
        input_string (str): String to be normalized

    Returns:
        normalized_string (str)
    """
    # Fold full-width / compatibility characters onto their canonical forms.
    result = unicodedata.normalize("NFKC", input_string)

    # Remove zero-width / bidi / invisible formatting characters.
    result = _INVISIBLE_CHARACTERS_RE.sub("", result)

    # Convert to lowercase.
    result = result.lower()

    # Turn punctuation (including full-width punctuation) into word
    # separators rather than deleting it, which would join adjacent words.
    result = _NON_WORD_RE.sub(" ", result)

    # Collapse all whitespace (tabs, newlines, ideographic spaces, ...) to a
    # single space and trim leading/trailing spaces.
    normalized_string = _WHITESPACE_RE.sub(" ", result).strip()

    return normalized_string


def get_input_substrings(normalized_input: str, keyword_length: int) -> List[str]:
    """
    Iterate over the input string and get substrings which have same length as as the keywords string

    Args:
        normalized_input (str): Normalized input string
        keyword_length (int): The number of words in the injection string

    Returns:
        List of input substrings that have the same length as the number of keywords in injection string
    """
    words_in_input_string = normalized_input.split(" ")
    input_substrings = []
    number_of_substrings = len(words_in_input_string) - keyword_length + 1
    for i in range(number_of_substrings):
        input_substrings.append(" ".join(words_in_input_string[i : i + keyword_length]))

    return input_substrings


def get_compact_input_substrings(
    compact_input: str, compact_keyword_length: int
) -> List[str]:
    """
    Slide a window over the whitespace-free input string.

    Used by the compact matching pass, which defeats word splitting tricks
    such as ``i g n o r e`` or ``ig-nore`` by comparing text with every
    whitespace character removed.

    Args:
        compact_input (str): Normalized input with all whitespace removed.
        compact_keyword_length (int): Window length in characters.

    Returns:
        List of input substrings of ``compact_keyword_length`` characters.
    """
    if compact_keyword_length <= 0:
        return []
    number_of_substrings = len(compact_input) - compact_keyword_length + 1
    return [
        compact_input[i : i + compact_keyword_length]
        for i in range(max(number_of_substrings, 0))
    ]


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


def get_compact_matched_words_score(
    substring: str, keyword_parts: List[str], max_matched_words: int
) -> float:
    """
    Score a whitespace-free substring against the whitespace-free keyword.

    Each keyword word that appears in order inside the substring counts as a
    matched word, so splitting the letters of a word with spaces, hyphens or
    zero-width characters no longer lowers the score. Consecutive keyword
    words may only be separated by a couple of characters, which is enough
    room for obfuscation characters but prevents unrelated words from
    different parts of a benign sentence from being stitched together. Very
    short keyword words (fewer than 4 characters) and single-word
    coincidences inside ordinary text are ignored to keep benign queries
    from being flagged.

    Args:
        substring (str): Whitespace-free input substring.
        keyword_parts (List[str]): Individual words of the keyword.
        max_matched_words (int): Cap applied to the matched word count.

    Returns:
        float: Score between 0 and 1, same scale as the word-based scorer.
    """
    matched_words_count = 0
    cursor = 0
    previous_end = -1
    # Maximum number of characters allowed between two consecutive matched
    # keyword words (a removed space, hyphen or zero-width character).
    max_gap = 2
    for part in keyword_parts:
        # Skip short particles (e.g. "to", "do", "not") whose letters occur
        # naturally inside many benign words.
        if len(part) < 4:
            continue
        position = substring.find(part, cursor)
        if position == -1:
            continue
        if previous_end != -1 and position - previous_end > max_gap:
            break
        matched_words_count += 1
        previous_end = position + len(part)
        cursor = previous_end

    # A single shared content word appears in plenty of harmless text, so at
    # least two keyword words must be present before this pass contributes.
    if matched_words_count < 2:
        return 0

    base_score = 0.5 + 0.5 * min(matched_words_count / max_matched_words, 1)
    return base_score


def detect_prompt_injection_using_heuristic_on_input(input: str) -> float:
    highest_score = 0
    max_matched_words = 5

    all_injection_keywords_strings = generate_injection_keywords()
    normalized_input_string = normalize_string(input)
    compact_input_string = normalized_input_string.replace(" ", "")

    for keyword_string in all_injection_keywords_strings:
        normalized_keyword_string = normalize_string(keyword_string)
        keywords = normalized_keyword_string.split(" ")
        # Compact (whitespace-free) form of the keyword.
        compact_keyword_string = normalized_keyword_string.replace(" ", "")

        # --- Word-based pass: identical algorithm on SDK and API ---
        # Generate substrings of similar length (to keyword length) in the input string
        input_substrings = get_input_substrings(normalized_input_string, len(keywords))

        for substring in input_substrings:
            similarity_score = _dice_coefficient(
                substring, normalized_keyword_string
            )

            matched_word_score = get_matched_words_score(
                substring, keywords, max_matched_words
            )

            # Adjust the score using the similarity score
            adjusted_score = matched_word_score - similarity_score * (
                1 / (max_matched_words * 2)
            )

            if adjusted_score > highest_score:
                highest_score = adjusted_score

        # --- Compact pass: defeat spacing / splitting obfuscation ---
        compact_substrings = get_compact_input_substrings(
            compact_input_string, len(compact_keyword_string)
        )
        for compact_substring in compact_substrings:
            compact_similarity_score = _dice_coefficient(
                compact_substring, compact_keyword_string
            )
            compact_matched_word_score = get_compact_matched_words_score(
                compact_substring, keywords, max_matched_words
            )
            compact_adjusted_score = (
                compact_matched_word_score
                - compact_similarity_score * (1 / (max_matched_words * 2))
            )

            if compact_adjusted_score > highest_score:
                highest_score = compact_adjusted_score

    return highest_score
