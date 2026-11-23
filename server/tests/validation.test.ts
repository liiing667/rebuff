/* eslint-disable @typescript-eslint/no-unused-expressions */
import { describe, it } from "mocha";
import { expect } from "chai";
import {
  validateBoolean,
  validateDetectRequestBody,
  validateThreshold,
  ValidationError,
} from "../lib/validation";

describe("server validation", () => {
  describe("validateThreshold", () => {
    // Invalid values interleaved with equivalent legal representations.
    [
      { value: NaN, valid: false },
      { value: 0, valid: true },
      { value: Infinity, valid: false },
      { value: 1, valid: true },
      { value: -Infinity, valid: false },
      { value: 0.75, valid: true },
      { value: -0.1, valid: false },
      { value: 0.9, valid: true },
      { value: 1.1, valid: false },
      { value: null, valid: true },
      { value: undefined, valid: true },
      { value: "0.5", valid: false },
      { value: true, valid: false },
    ].forEach(({ value, valid }) => {
      it(`${valid ? "accepts" : "rejects"} ${String(value)}`, () => {
        if (valid) {
          expect(() => validateThreshold(value, "t")).to.not.throw();
        } else {
          expect(() => validateThreshold(value, "t")).to.throw(ValidationError);
        }
      });
    });
  });

  describe("validateBoolean", () => {
    it("accepts booleans and rejects coercible values", () => {
      expect(() => validateBoolean(true, "f")).to.not.throw();
      expect(() => validateBoolean(false, "f")).to.not.throw();
      ["true", 1, 0, null, undefined].forEach((v) =>
        expect(() => validateBoolean(v, "f")).to.throw(ValidationError)
      );
    });
  });

  describe("validateDetectRequestBody", () => {
    it("fills defaults for an empty body", () => {
      const normalized = validateDetectRequestBody({});
      expect(normalized.runHeuristicCheck).to.equal(true);
      expect(normalized.runVectorCheck).to.equal(true);
      expect(normalized.runLanguageModelCheck).to.equal(true);
      expect(normalized.maxHeuristicScore).to.equal(null);
    });

    it("rejects a NaN threshold before detection", () => {
      expect(() =>
        validateDetectRequestBody({ maxHeuristicScore: NaN })
      ).to.throw(ValidationError, /finite/);
    });

    it("rejects an Infinity threshold before detection", () => {
      expect(() =>
        validateDetectRequestBody({ maxVectorScore: Infinity })
      ).to.throw(ValidationError, /finite/);
    });

    it("rejects a negative threshold", () => {
      expect(() =>
        validateDetectRequestBody({ maxModelScore: -0.01 })
      ).to.throw(ValidationError, /between 0 and 1/);
    });

    it("accepts the 0 and 1 boundaries", () => {
      const normalized = validateDetectRequestBody({
        maxHeuristicScore: 0,
        maxVectorScore: 1,
      });
      expect(normalized.maxHeuristicScore).to.equal(0);
      expect(normalized.maxVectorScore).to.equal(1);
    });

    it("rejects a non-boolean run flag", () => {
      expect(() =>
        validateDetectRequestBody({ runHeuristicCheck: "false" })
      ).to.throw(ValidationError, /boolean/);
    });

    it("rejects a non-string userInputBase64", () => {
      expect(() =>
        validateDetectRequestBody({ userInputBase64: 123 })
      ).to.throw(ValidationError, /userInputBase64/);
    });
  });
});
