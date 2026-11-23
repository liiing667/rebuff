import crypto from "crypto";

export const DEFAULT_CANARY_FORMAT = "<!-- {canary_word} -->";
export const CANARY_WORD_PLACEHOLDER = "{canary_word}";
export const MIN_CANARY_LENGTH = 1;
const MAX_REMEMBERED_CANARIES = 2048;

// Keep the exact marker for each identity produced by `embedCanary` so a
// later completion check can use the same identity even when a custom format
// places the word directly next to other word characters. The store is
// bounded so a long-running process does not retain every canary forever.
const canaryMarkers = new Map<string, Set<string>>();

function rememberCanaryMarker(canaryWord: string, marker: string): void {
  if (!canaryWord || !marker) {
    return;
  }

  const markers = canaryMarkers.get(canaryWord) ?? new Set<string>();
  markers.add(marker);
  canaryMarkers.delete(canaryWord);
  canaryMarkers.set(canaryWord, markers);

  while (canaryMarkers.size > MAX_REMEMBERED_CANARIES) {
    const oldest = canaryMarkers.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    canaryMarkers.delete(oldest);
  }
}

function hasRememberedMarker(
  completion: string,
  canaryWord: string
): boolean {
  const markers = canaryMarkers.get(canaryWord);
  if (!markers) {
    return false;
  }
  return Array.from(markers).some((marker) => completion.includes(marker));
}

export interface PreparedCanary {
  // The identity that must be used for completion checks and logging.
  canaryWord: string;
  // The exact text embedded in the prompt.
  marker: string;
  // The prompt with the marker embedded.
  prompt: string;
}

/**
 * Generate a hexadecimal canary word of exactly `length` characters.
 *
 * `crypto.randomBytes` takes a byte count and returns two hex characters per
 * byte. Using `length / 2` directly drops a character for odd lengths and
 * throws for non-integers; rounding up and slicing keeps the requested size.
 */
export function generateCanaryWord(length = 8): string {
  if (!Number.isInteger(length) || length < MIN_CANARY_LENGTH) {
    throw new RangeError(
      `canary word length must be an integer of at least ${MIN_CANARY_LENGTH}`
    );
  }

  return crypto
    .randomBytes(Math.ceil(length / 2))
    .toString("hex")
    .slice(0, length);
}

export function formatCanaryMarker(
  canaryWord: string,
  canaryFormat: string = DEFAULT_CANARY_FORMAT
): string {
  return canaryFormat.split(CANARY_WORD_PLACEHOLDER).join(canaryWord);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Recover the identity of a canary that is already embedded in a legacy
 * prompt, so callers do not accidentally add a second canary.
 */
export function extractCanaryWord(
  prompt: string,
  canaryFormat: string = DEFAULT_CANARY_FORMAT
): string | undefined {
  if (prompt.includes(CANARY_WORD_PLACEHOLDER)) {
    return undefined;
  }

  const placeholderIndex = canaryFormat.indexOf(CANARY_WORD_PLACEHOLDER);
  if (placeholderIndex < 0) {
    return undefined;
  }

  const before = canaryFormat.slice(0, placeholderIndex);
  const after = canaryFormat.slice(
    placeholderIndex + CANARY_WORD_PLACEHOLDER.length
  );
  if (!before && !after) {
    // A bare word is indistinguishable from arbitrary prompt text.
    return undefined;
  }
  const pattern = new RegExp(
    `${escapeRegExp(before)}([\\s\\S]+?)${escapeRegExp(after)}`
  );
  const match = prompt.match(pattern);
  if (!match) {
    return undefined;
  }

  const canaryWord = match[1].trim();
  if (!canaryWord || /\s/.test(canaryWord)) {
    return undefined;
  }
  return canaryWord;
}

function looksLikeMarker(value: string): boolean {
  return Array.from(value).some(
    (character) =>
      /\s/.test(character) || "<>{}[]()\"'`".includes(character)
  );
}

function isWordCharacter(character: string | undefined): boolean {
  return !!character && /[A-Za-z0-9_]/.test(character);
}

function containsStandaloneToken(text: string, token: string): boolean {
  if (!token) {
    return false;
  }

  let fromIndex = 0;
  while (fromIndex <= text.length - token.length) {
    const index = text.indexOf(token, fromIndex);
    if (index < 0) {
      return false;
    }

    const before = index > 0 ? text[index - 1] : undefined;
    const afterIndex = index + token.length;
    const after = afterIndex < text.length ? text[afterIndex] : undefined;
    if (!isWordCharacter(before) && !isWordCharacter(after)) {
      return true;
    }
    fromIndex = index + 1;
  }
  return false;
}

/**
 * Check a completion against the same identity that was embedded. A bare
 * canary must stand alone, so `dead` does not match `deadline` and a canary
 * does not match inside a longer hash.
 */
export function isCanaryLeaked(
  completion: string,
  canaryWord: string,
  canaryFormat?: string
): boolean {
  if (
    !completion ||
    typeof completion !== "string" ||
    !canaryWord ||
    typeof canaryWord !== "string"
  ) {
    return false;
  }

  if (hasRememberedMarker(completion, canaryWord)) {
    return true;
  }

  if (canaryFormat !== undefined) {
    const marker = formatCanaryMarker(canaryWord, canaryFormat);
    if (marker && marker !== canaryWord && completion.includes(marker)) {
      return true;
    }
  }

  if (looksLikeMarker(canaryWord)) {
    return completion.includes(canaryWord);
  }
  return containsStandaloneToken(completion, canaryWord);
}

/**
 * Embed a canary while preserving explicit markers and legacy prompt
 * templates. A `{canary_word}` placeholder is filled in place; an already
 * embedded marker is reused; otherwise the marker is prepended once.
 */
export function embedCanary(
  prompt: string,
  canaryWord?: string,
  canaryFormat: string = DEFAULT_CANARY_FORMAT
): PreparedCanary {
  const hasPlaceholder = canaryFormat.includes(CANARY_WORD_PLACEHOLDER);
  let identity = canaryWord;
  if (identity !== undefined && identity !== null) {
    // Accept an already formatted explicit marker as well as a bare word.
    const explicitWord = extractCanaryWord(identity, canaryFormat);
    if (explicitWord !== undefined) {
      identity = explicitWord;
    }
  }
  const existing = extractCanaryWord(prompt, canaryFormat);

  if (identity === undefined || identity === null) {
    if (existing !== undefined) {
      identity = existing;
    } else if (!hasPlaceholder && canaryFormat) {
      identity = canaryFormat;
    } else {
      identity = generateCanaryWord();
    }
  }

  if (typeof identity !== "string" || identity.length === 0) {
    throw new Error("canaryWord must be a non-empty string");
  }

  const marker = hasPlaceholder
    ? formatCanaryMarker(identity, canaryFormat)
    : canaryFormat || identity;
  const canaryIdentity = hasPlaceholder ? identity : marker;

  let promptWithCanary: string;
  if (prompt.includes(CANARY_WORD_PLACEHOLDER)) {
    promptWithCanary = prompt
      .split(CANARY_WORD_PLACEHOLDER)
      .join(identity);
  } else if (existing !== undefined && existing !== identity && hasPlaceholder) {
    const oldMarker = formatCanaryMarker(existing, canaryFormat);
    promptWithCanary = prompt.split(oldMarker).join(marker);
  } else if (marker && prompt.includes(marker)) {
    promptWithCanary = prompt;
  } else if (marker) {
    promptWithCanary = `${marker}\n${prompt}`;
  } else {
    promptWithCanary = prompt;
  }

  rememberCanaryMarker(canaryIdentity, marker);
  return {
    canaryWord: canaryIdentity,
    marker,
    prompt: promptWithCanary,
  };
}
