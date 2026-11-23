import { RebuffError, TacticName, TacticOverride } from "./interface";

/**
 * Thresholds are scores on a 0..1 scale. Both boundaries are valid.
 * `null`/`undefined` means "use the built-in default threshold" and is allowed.
 */
export function validateThreshold(
  value: unknown,
  parameterName: string
): asserts value is number | null | undefined {
  if (value === null || value === undefined) {
    return;
  }
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new RebuffError(
      `${parameterName} must be a finite number between 0 and 1`
    );
  }
  if (!Number.isFinite(value)) {
    throw new RebuffError(
      `${parameterName} must be finite (Infinity and -Infinity are not allowed)`
    );
  }
  if (value < 0 || value > 1) {
    throw new RebuffError(
      `${parameterName} must be between 0 and 1 (inclusive)`
    );
  }
}

/**
 * Strict boolean check. A request flag must actually be a boolean; values such
 * as the strings "true"/"false", numbers or null are rejected instead of being
 * silently coerced.
 */
export function validateBoolean(
  value: unknown,
  parameterName: string
): asserts value is boolean {
  if (typeof value !== "boolean") {
    throw new RebuffError(`${parameterName} must be a boolean`);
  }
}

const VALID_TACTIC_NAMES: ReadonlySet<string> = new Set<string>(
  Object.values(TacticName)
);

/**
 * Validates caller-supplied tactic overrides before any tactic is executed.
 * Unknown tactic names (which would otherwise be silently ignored), duplicate
 * entries and invalid thresholds are all rejected.
 */
export function validateTacticOverrides(
  tacticOverrides: unknown
): asserts tacticOverrides is TacticOverride[] {
  if (tacticOverrides === undefined) {
    return;
  }
  if (!Array.isArray(tacticOverrides)) {
    throw new RebuffError("tacticOverrides must be an array");
  }

  const seenNames = new Set<string>();
  for (const override of tacticOverrides) {
    if (
      override === null ||
      typeof override !== "object" ||
      !VALID_TACTIC_NAMES.has((override as TacticOverride).name)
    ) {
      throw new RebuffError(
        `tacticOverrides contains an unknown tactic name; expected one of: ${[
          ...VALID_TACTIC_NAMES,
        ].join(", ")}`
      );
    }

    const { name, threshold, run } = override as TacticOverride;

    if (seenNames.has(name)) {
      throw new RebuffError(
        `tacticOverrides contains a duplicate entry for tactic "${name}"`
      );
    }
    seenNames.add(name);

    validateThreshold(threshold, `tacticOverrides[${name}].threshold`);

    if (run !== undefined) {
      validateBoolean(run, `tacticOverrides[${name}].run`);
    }
  }
}
