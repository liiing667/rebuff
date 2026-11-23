/**
 * Request validation for the detect endpoints.
 *
 * A plain `typeof x === "number"` check is not enough because NaN is also a
 * number and every comparison against NaN (or Infinity) silently evaluates to
 * false, which lets malicious or malformed thresholds bypass detection. These
 * helpers reject such values before any detection logic runs, while keeping
 * the 0 and 1 boundaries and the default thresholds valid.
 */

export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

/**
 * Validates a detection threshold. `null`/`undefined` means the caller wants
 * the server-side default and is permitted.
 */
export function validateThreshold(
  value: unknown,
  parameterName: string
): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new ValidationError(
      `${parameterName} must be a finite number between 0 and 1`
    );
  }
  if (!Number.isFinite(value)) {
    throw new ValidationError(
      `${parameterName} must be finite (Infinity and -Infinity are not allowed)`
    );
  }
  if (value < 0 || value > 1) {
    throw new ValidationError(
      `${parameterName} must be between 0 and 1 (inclusive)`
    );
  }
  return value;
}

/**
 * Strict boolean validation. Strings like "true"/"false" and other truthy
 * values are rejected instead of being silently coerced.
 */
export function validateBoolean(
  value: unknown,
  parameterName: string
): boolean {
  if (typeof value !== "boolean") {
    throw new ValidationError(`${parameterName} must be a boolean`);
  }
  return value;
}

export interface DetectRequestBody {
  userInputBase64?: unknown;
  runHeuristicCheck?: unknown;
  runVectorCheck?: unknown;
  runLanguageModelCheck?: unknown;
  maxHeuristicScore?: unknown;
  maxModelScore?: unknown;
  maxVectorScore?: unknown;
}

/**
 * Validates and normalizes the full body of a /api/detect request. Throws a
 * ValidationError (mapped to HTTP 400 by the caller) when any field is invalid.
 */
export function validateDetectRequestBody(body: DetectRequestBody) {
  if (
    body.userInputBase64 !== undefined &&
    body.userInputBase64 !== null &&
    typeof body.userInputBase64 !== "string"
  ) {
    throw new ValidationError("userInputBase64 must be a string");
  }

  return {
    runHeuristicCheck: validateBoolean(
      body.runHeuristicCheck ?? true,
      "runHeuristicCheck"
    ),
    runVectorCheck: validateBoolean(
      body.runVectorCheck ?? true,
      "runVectorCheck"
    ),
    runLanguageModelCheck: validateBoolean(
      body.runLanguageModelCheck ?? true,
      "runLanguageModelCheck"
    ),
    maxHeuristicScore: validateThreshold(
      body.maxHeuristicScore ?? null,
      "maxHeuristicScore"
    ),
    maxModelScore: validateThreshold(
      body.maxModelScore ?? null,
      "maxModelScore"
    ),
    maxVectorScore: validateThreshold(
      body.maxVectorScore ?? null,
      "maxVectorScore"
    ),
  };
}
