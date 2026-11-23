import stringSimilarity from "string-similarity";
import { normalizeString } from "../lib/prompts";
import { TacticName } from "../interface";
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

const MAX_MATCHED_WORDS = 5;

// Score a whitespace-free substring against the whitespace-free keyword.
// Each keyword word that appears in order inside the substring counts as a
// matched word, so splitting the letters of a word with spaces, hyphens or
// zero-width characters no longer lowers the score. Consecutive keyword
// words may only be separated by a couple of characters, which is enough
// room for obfuscation characters but prevents unrelated words from
// different parts of a benign sentence from being stitched together. Short
// keyword words and single-word coincidences are ignored to avoid flagging
// ordinary text.
function getCompactMatchedWordsScore(
  substring: string,
  keywordParts: string[],
  maxMatchedWords: number
): number {
  let matchedWordsCount = 0;
  let cursor = 0;
  let previousEnd = -1;
  const maxGap = 2;

  for (const part of keywordParts) {
    // Skip short particles (e.g. "to", "do", "not") whose letters occur
    // naturally inside many benign words.
    if (part.length < 4) {
      continue;
    }
    const position = substring.indexOf(part, cursor);
    if (position === -1) {
      continue;
    }
    if (previousEnd !== -1 && position - previousEnd > maxGap) {
      break;
    }
    matchedWordsCount += 1;
    previousEnd = position + part.length;
    cursor = previousEnd;
  }

  // A single shared content word appears in plenty of harmless text, so at
  // least two keyword words must be present before this pass contributes.
  if (matchedWordsCount < 2) {
    return 0;
  }

  return 0.5 + 0.5 * Math.min(matchedWordsCount / maxMatchedWords, 1);
}

export default class Heuristic implements Tactic {
  name = TacticName.Heuristic;
  defaultThreshold: number;

  constructor(threshold: number) {
    this.defaultThreshold = threshold;
  }

  execute(input: string): Promise<TacticExecution> {
    let highestScore = 0;
    const normalizedInput = normalizeString(input);
    // Whitespace-free view of the input used to detect letter-level word
    // splitting such as "i g n o r e ..." or "ig-nore ...".
    const compactInput = normalizedInput.split(" ").join("");

    for (const keyword of injectionKeywords) {
      const normalizedKeyword = normalizeString(keyword);
      const keywordParts = normalizedKeyword.split(" ");
      const keywordLength = keywordParts.length;
      const compactKeyword = normalizedKeyword.split(" ").join("");

      // Generate substrings of similar length in the input string
      const inputParts = normalizedInput.split(" ");
      const inputSubstrings = [];
      for (let i = 0; i <= inputParts.length - keywordLength; i++) {
        inputSubstrings.push(inputParts.slice(i, i + keywordLength).join(" "));
      }

      // Calculate the similarity score between the keyword and each substring
      for (const substring of inputSubstrings) {
        const similarityScore = stringSimilarity.compareTwoStrings(
          normalizedKeyword,
          substring
        );

        // Calculate the score based on the number of consecutive words matched
        const matchedWordsCount = keywordParts.filter(
          (part, index) => substring.split(" ")[index] === part
        ).length;
        const baseScore =
          matchedWordsCount > 0
            ? 0.5 +
              0.5 * Math.min(matchedWordsCount / MAX_MATCHED_WORDS, 1)
            : 0;

        // Adjust the score using the similarity score
        const adjustedScore =
          baseScore - similarityScore * (1 / (MAX_MATCHED_WORDS * 2));

        // Update the highest score if the current adjusted score is higher
        if (adjustedScore > highestScore) {
          highestScore = adjustedScore;
        }
      }

      // Compact pass: slide a window over the whitespace-free input so
      // attacks whose words have been split apart letter by letter still
      // receive the same score as the unsplit phrase.
      for (let i = 0; i <= compactInput.length - compactKeyword.length; i++) {
        const compactSubstring = compactInput.slice(
          i,
          i + compactKeyword.length
        );
        const compactSimilarityScore = stringSimilarity.compareTwoStrings(
          compactKeyword,
          compactSubstring
        );
        const compactBaseScore = getCompactMatchedWordsScore(
          compactSubstring,
          keywordParts,
          MAX_MATCHED_WORDS
        );
        const compactAdjustedScore =
          compactBaseScore -
          compactSimilarityScore * (1 / (MAX_MATCHED_WORDS * 2));

        if (compactAdjustedScore > highestScore) {
          highestScore = compactAdjustedScore;
        }
      }
    }
    
    return Promise.resolve({ score: highestScore });
  }
}
