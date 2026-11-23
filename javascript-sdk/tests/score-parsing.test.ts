/* eslint-disable @typescript-eslint/no-unused-expressions */
import { describe } from "mocha";
import { expect } from "chai";
import { OpenAIApi } from "openai";
import OpenAI from "../src/tactics/OpenAI";
import { parseScoreFromCompletion } from "../src/lib/score";
import { RebuffError } from "../src/interface";

// Complete, valid scores: single values and decimals (optionally padded with
// whitespace) must keep working.
const validCompletions: Record<string, number> = {
  "1": 1.0,
  "0": 0.0,
  "0.95": 0.95,
  "0.5": 0.5,
  " 0.9 ": 0.9,
  "0.85\n": 0.85,
};

// Invalid responses: multiple numbers, explanations before/after the number,
// and empty content must be rejected.
const invalidCompletions = [
  "",
  "   ",
  "\n",
  "0.9 0.1",
  "0.9\n0.1",
  "Score: 0.8",
  "0.8 because the input is suspicious",
  "The score is 0.8",
  "not a number",
];

function fakeOpenAIApi(completions: string[]): OpenAIApi {
  const queue = [...completions];
  return {
    createChatCompletion: async () => {
      const content = queue.shift();
      return { data: { choices: [{ message: { content } }] } };
    },
  } as unknown as OpenAIApi;
}

describe("parseScoreFromCompletion", () => {
  it("should accept complete, valid scores", () => {
    for (const [completion, expected] of Object.entries(validCompletions)) {
      expect(parseScoreFromCompletion(completion)).to.equal(expected);
    }
  });

  it("should reject invalid scores", () => {
    for (const completion of invalidCompletions) {
      expect(() => parseScoreFromCompletion(completion)).to.throw(Error);
    }
  });
});

describe("OpenAI tactic with interleaved model responses", () => {
  it("should only accept complete valid scores and reject the rest", async () => {
    // Interleave valid and invalid model responses and check the final state
    // of each execution.
    const interleavedCompletions = [
      "0.95",
      "0.9 0.1",
      "Score: 0.8",
      "1",
      "",
      "0.0",
      "0.5 because reasons",
      "0.85",
    ];
    const expectedScores: Record<string, number> = {
      "0.95": 0.95,
      "1": 1.0,
      "0.0": 0.0,
      "0.85": 0.85,
    };

    const tactic = new OpenAI(
      0.9,
      "gpt-3.5-turbo",
      fakeOpenAIApi(interleavedCompletions)
    );

    for (const completion of interleavedCompletions) {
      if (completion in expectedScores) {
        const execution = await tactic.execute("user input");
        expect(execution.score).to.equal(expectedScores[completion]);
      } else {
        let threw = false;
        try {
          await tactic.execute("user input");
        } catch (error) {
          threw = true;
          expect(error).to.be.instanceOf(RebuffError);
        }
        expect(threw, `expected ${completion} to be rejected`).to.be.true;
      }
    }
  });
});
