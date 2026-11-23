// A complete, valid score is a single floating point number and nothing else
// (leading/trailing whitespace is tolerated). Responses that contain multiple
// numbers, explanatory text before or after the number, or no number at all
// are rejected.
const SCORE_PATTERN = /^[+-]?(\d+(\.\d*)?|\.\d+)$/;

/**
 * Parses the score returned by the language model.
 *
 * The completion must consist of exactly one valid floating point number and
 * nothing else (surrounding whitespace is ignored).
 *
 * @param completion The raw completion returned by the language model.
 * @returns The parsed score.
 * @throws Error If the completion is empty, contains multiple numbers, or
 * contains anything other than a single floating point number.
 */
export function parseScoreFromCompletion(completion: string): number {
  const text = (completion ?? "").trim();
  if (text === "") {
    throw new Error("Language model returned an empty completion");
  }
  if (!SCORE_PATTERN.test(text)) {
    throw new Error(`Language model returned an invalid score: ${completion}`);
  }
  return parseFloat(text);
}
