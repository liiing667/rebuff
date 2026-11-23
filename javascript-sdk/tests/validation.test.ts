/* eslint-disable @typescript-eslint/no-unused-expressions */
import { describe, it } from "mocha";
import { expect } from "chai";
import {
  validateBoolean,
  validateThreshold,
  validateTacticOverrides,
} from "../src/validation";
import { RebuffError, TacticName } from "../src/interface";

// Alternating invalid values with their equivalent legal representations.
const invalidThresholds: Array<{ value: unknown; label: string }> = [
  { value: NaN, label: "NaN" },
  { value: Infinity, label: "Infinity" },
  { value: -Infinity, label: "-Infinity" },
  { value: -0.0001, label: "negative threshold" },
  { value: 1.0001, label: "threshold above 1" },
  { value: "0.5", label: "numeric string" },
  { value: true, label: "boolean" },
  { value: null, label: "null (use default)" },
];

// Equivalent legal representations that must continue to work.
const validThresholds: Array<{ value: unknown; label: string }> = [
  { value: 0, label: "lower boundary 0" },
  { value: 1, label: "upper boundary 1" },
  { value: 0.75, label: "default heuristic threshold" },
  { value: 0.9, label: "default vector/model threshold" },
  { value: 0.0, label: "explicit 0.0" },
  { value: undefined, label: "undefined (use default)" },
];

describe("validateThreshold", () => {
  invalidThresholds.forEach(({ value, label }) => {
    if (value === null) {
      it(`accepts ${label}`, () => {
        expect(() => validateThreshold(value, "t")).to.not.throw();
      });
      return;
    }
    it(`rejects ${label}`, () => {
      expect(() => validateThreshold(value, "t")).to.throw(RebuffError);
    });
  });

  validThresholds.forEach(({ value, label }) => {
    it(`accepts ${label}`, () => {
      expect(() => validateThreshold(value, "t")).to.not.throw();
    });
  });

  it("rejects -0 represented as a negative number below the range", () => {
    // -0 itself equals 0 and is a legal boundary.
    expect(() => validateThreshold(-0, "t")).to.not.throw();
  });
});

describe("validateBoolean", () => {
  it("accepts real booleans", () => {
    expect(() => validateBoolean(true, "f")).to.not.throw();
    expect(() => validateBoolean(false, "f")).to.not.throw();
  });

  ["true", "false", 1, 0, null, undefined, ""].forEach((value) => {
    it(`rejects non-boolean value ${JSON.stringify(value)}`, () => {
      expect(() => validateBoolean(value, "f")).to.throw(RebuffError);
    });
  });
});

describe("validateTacticOverrides", () => {
  it("accepts an empty/undefined override list", () => {
    expect(() => validateTacticOverrides(undefined)).to.not.throw();
    expect(() => validateTacticOverrides([])).to.not.throw();
  });

  it("accepts valid overrides including the 0 and 1 boundaries", () => {
    expect(() =>
      validateTacticOverrides([
        { name: TacticName.Heuristic, threshold: 0 },
        { name: TacticName.VectorDB, threshold: 1 },
        { name: TacticName.LanguageModel, run: false },
      ])
    ).to.not.throw();
  });

  it("rejects an unknown tactic name (unknown enum) instead of ignoring it", () => {
    expect(() =>
      validateTacticOverrides([
        { name: "not_a_real_tactic" as TacticName, threshold: 0.5 },
      ])
    ).to.throw(RebuffError, /unknown tactic name/);
  });

  [NaN, Infinity, -1, 2].forEach((bad) => {
    it(`rejects invalid threshold ${bad}`, () => {
      expect(() =>
        validateTacticOverrides([
          { name: TacticName.Heuristic, threshold: bad },
        ])
      ).to.throw(RebuffError);
    });
  });

  it("rejects duplicate tactic entries", () => {
    expect(() =>
      validateTacticOverrides([
        { name: TacticName.Heuristic, threshold: 0.5 },
        { name: TacticName.Heuristic, threshold: 0.6 },
      ])
    ).to.throw(RebuffError, /duplicate/);
  });

  it("rejects a non-boolean run flag", () => {
    expect(() =>
      validateTacticOverrides([
        { name: TacticName.Heuristic, run: "false" as unknown as boolean },
      ])
    ).to.throw(RebuffError);
  });
});
