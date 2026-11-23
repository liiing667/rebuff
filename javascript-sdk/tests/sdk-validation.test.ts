/* eslint-disable @typescript-eslint/no-unused-expressions */
import { describe, it } from "mocha";
import { expect } from "chai";
import RebuffSdk from "../src/sdk";
import { RebuffError, TacticName } from "../src/interface";

// The constructor (without init) never touches OpenAI/Pinecone, so these tests
// stay fully offline. Validation runs before getStrategies() is reached.
const sdk = new RebuffSdk({
  openai: { apikey: "test-key", model: "gpt-3.5-turbo" },
  vectorDB: {
    pinecone: {
      environment: "test-env",
      apikey: "test-key",
      index: "test-index",
    },
  },
});

describe("RebuffSdk.detectInjection input validation", () => {
  const invalidThresholds: Array<[unknown, string]> = [
    [NaN, "NaN"],
    [Infinity, "Infinity"],
    [-Infinity, "-Infinity"],
    [-0.5, "negative threshold"],
    [1.5, "threshold above 1"],
  ];

  invalidThresholds.forEach(([bad, label]) => {
    it(`rejects ${label} before detection runs`, async () => {
      try {
        await sdk.detectInjection({
          userInput: "hello",
          tacticOverrides: [
            { name: TacticName.Heuristic, threshold: bad as number },
          ],
        });
        expect.fail("expected detectInjection to throw");
      } catch (error) {
        expect(error).to.be.instanceOf(RebuffError);
      }
    });
  });

  it("rejects an unknown tactic name (unknown enum)", async () => {
    try {
      await sdk.detectInjection({
        userInput: "hello",
        tacticOverrides: [
          { name: "made_up_tactic" as TacticName, threshold: 0.5 },
        ],
      });
      expect.fail("expected detectInjection to throw");
    } catch (error) {
      expect(error).to.be.instanceOf(RebuffError);
      expect((error as RebuffError).message).to.match(/unknown tactic name/);
    }
  });

  // Equivalent legal representations interleaved with the invalid cases above.
  [0, 1, 0.75].forEach((good) => {
    it(`accepts the valid boundary/default threshold ${good} past validation`, async () => {
      // Validation passes; the call only fails afterwards when it tries to
      // initialize external stores, proving the threshold itself was accepted.
      try {
        await sdk.detectInjection({
          userInput: "hello",
          tacticOverrides: [
            { name: TacticName.Heuristic, threshold: good },
            { name: TacticName.LanguageModel, run: false },
            { name: TacticName.VectorDB, run: false },
          ],
        });
      } catch (error) {
        expect(error).to.not.be.instanceOf(RebuffError);
        expect((error as Error).message).to.not.match(/threshold|tactic/);
      }
    });
  });

  it("still requires userInput", async () => {
    try {
      await sdk.detectInjection({ userInput: "" });
      expect.fail("expected detectInjection to throw");
    } catch (error) {
      expect(error).to.be.instanceOf(RebuffError);
      expect((error as RebuffError).message).to.match(/userInput is required/);
    }
  });
});
