import stringSimilarity from "string-similarity";
import { normalizeString } from "../lib/prompts";
import { MAX_INPUT_LENGTH, TacticName } from "../interface";
import Tactic, { TacticExecution } from "./Tactic";

function generateInjectionKeywords() {
  // Define lists of verbs, adjectives, prepositions, and objects
  const verbs = [
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
  ];

  const adjectives = [
    "",
    "prior",
    "previous",
    "preceding",
    "above",
    "foregoing",
    "earlier",
    "initial",
  ];

  const prepositions = [
    "",
    "and start over",
    "and start anew",
    "and begin afresh",
    "and start from scratch",
  ];

  const objects = [
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
  ];

  // Generate all possible combinations of sentences
  const injectionKeywords = [];
  for (const verb of verbs) {
    for (const adjective of adjectives) {
      for (const object of objects) {
          for (const preposition of prepositions) {
          injectionKeywords.push(
              `${verb} ${adjective} ${object} ${preposition}`.trim()
          );
          }
      }
    }
  }

  return injectionKeywords;
}

// Generate and print the injection keywords
const injectionKeywords = generateInjectionKeywords();

// Precompute the normalized keywords and their word parts once at module load,
// instead of recomputing them for every scored substring.
const normalizedInjectionKeywords: Array<[string, string[]]> =
  injectionKeywords.map((keyword) => {
    const normalizedKeyword = normalizeString(keyword);
    return [normalizedKeyword, normalizedKeyword.split(" ")];
  });

export default class Heuristic implements Tactic {
  name = TacticName.Heuristic;
  defaultThreshold: number;

  constructor(threshold: number) {
    this.defaultThreshold = threshold;
  }

  execute(
    input: string,
    thresholdOverride?: number,
    maxInputLength: number = MAX_INPUT_LENGTH
  ): Promise<TacticExecution> {
    let highestScore = 0;

    // Bound scoring to the shared input budget so huge inputs cannot stall
    // the heuristic check. All state is local to this call, so a retried
    // execution never continues with fragments from a previous attempt.
    const budgetedInput =
      input.length > maxInputLength ? input.slice(0, maxInputLength) : input;
    const normalizedInput = normalizeString(budgetedInput);
    const inputParts = normalizedInput.split(" ");
    const numInputParts = inputParts.length;
    const similarityCache = new Map<string, number>();

    for (let k = 0; k < normalizedInjectionKeywords.length; k++) {
      const [normalizedKeyword, keywordParts] = normalizedInjectionKeywords[k];
      const keywordLength = keywordParts.length;
      if (keywordLength > numInputParts) {
        continue;
      }

      // Generate substrings of similar length in the input string
      for (let i = 0; i <= numInputParts - keywordLength; i++) {
        // Calculate the score based on the number of consecutive words matched
        let matchedWordsCount = 0;
        for (let j = 0; j < keywordLength; j++) {
          if (inputParts[i + j] === keywordParts[j]) {
            matchedWordsCount++;
          }
        }

        if (matchedWordsCount === 0) {
          // The adjusted score would be <= 0 here, so it can never raise the
          // highest score. Skipping the expensive similarity call keeps long
          // documents fast without changing the result.
          continue;
        }

        const substring = inputParts.slice(i, i + keywordLength).join(" ");

        // Calculate the similarity score between the keyword and each substring.
        // Repeated windows (common in padded or repetitive inputs) are only
        // scored once thanks to the call-local cache.
        const cacheKey = `${k}:${substring}`;
        let similarityScore = similarityCache.get(cacheKey);
        if (similarityScore === undefined) {
          similarityScore = stringSimilarity.compareTwoStrings(
            normalizedKeyword,
            substring
          );
          similarityCache.set(cacheKey, similarityScore);
        }

        const maxMatchedWords = 5;
        const baseScore =
          0.5 + 0.5 * Math.min(matchedWordsCount / maxMatchedWords, 1);

        // Adjust the score using the similarity score
        const adjustedScore =
          baseScore - similarityScore * (1 / (maxMatchedWords * 2));

        // Update the highest score if the current adjusted score is higher
        if (adjustedScore > highestScore) {
          highestScore = adjustedScore;
        }
      }
    }
    
    return Promise.resolve({ score: highestScore });
  }
}
