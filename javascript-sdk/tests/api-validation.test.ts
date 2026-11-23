/* eslint-disable @typescript-eslint/no-unused-expressions */
import { describe, it } from "mocha";
import { expect } from "chai";
import RebuffApi from "../src/api";
import { RebuffError } from "../src/interface";

describe("RebuffApi.detectInjection", () => {
  const api = new RebuffApi({ apiKey: "test", apiUrl: "http://localhost" });

  // Invalid parameters must be rejected before any network activity. These
  // cases never reach fetch, so they run fully offline.
  [NaN, Infinity, -Infinity, -0.1, 1.1].forEach((bad) => {
    it(`rejects invalid threshold ${bad} before making a request`, async () => {
      try {
        await api.detectInjection({
          userInput: "hello",
          maxHeuristicScore: bad,
        } as any);
        expect.fail("expected a RebuffError");
      } catch (error) {
        expect(error).to.be.instanceOf(RebuffError);
      }
    });
  });

  it("rejects a non-boolean run flag before making a request", async () => {
    try {
      await api.detectInjection({
        userInput: "hello",
        runHeuristicCheck: "false" as unknown as boolean,
      } as any);
      expect.fail("expected a RebuffError");
    } catch (error) {
      expect(error).to.be.instanceOf(RebuffError);
    }
  });

  // Equivalent legal representations interleaved with the invalid cases above.
  [0, 1, 0.75, 0.9].forEach((good) => {
    it(`accepts the valid boundary/default threshold ${good} past validation`, async () => {
      // Validation passes; only the (offline-unreachable) network call can fail
      // afterwards, proving the threshold itself was accepted.
      try {
        await api.detectInjection({
          userInput: "hello",
          maxHeuristicScore: good,
          maxVectorScore: good,
          maxModelScore: good,
        } as any);
      } catch (error) {
        expect((error as RebuffError).message).to.not.match(
          /finite|between 0 and 1|boolean/
        );
      }
    });
  });
});
