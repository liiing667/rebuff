export enum TacticName {
  // A series of heuristics are used to determine whether the input is prompt injection.
  Heuristic = "heuristic",
  // A language model is asked if the input appears to be prompt injection.
  LanguageModel = "language_model",
  // A vector database of known prompt injection attacks is queried for similarity.
  VectorDB = "vector_db",
}

export interface TacticOverride {
  // The name of the tactic to override.
  name: TacticName;
  // The threshold to use for this tactic. If the score is above this threshold, the tactic will be considered detected.
  // If not specified, the default threshold for the tactic will be used.
  threshold?: number;
  // Whether to run this tactic. Defaults to true if not specified.
  run?: boolean;
}

export interface DetectRequest {
  // The user input to check for prompt injection.
  userInput: string;
  // The base64-encoded user input. If this is specified, the user input will be ignored.
  userInputBase64?: string;
  // Any tactics to change behavior for. If any tactic is not specified, the default threshold for that tactic will be
  // used.
  tacticOverrides?: TacticOverride[];
}

export interface TacticResult {
  // The name of the tactic.
  name: TacticName;
  // The score for the tactic. This is a number between 0 and 1. The closer to 1, the more likely that this is a
  // prompt injection attempt.
  score: number;
  // Whether this tactic evaluated the input as a prompt injection attempt.
  detected: boolean;
  // The threshold used for this tactic. If the score is above this threshold, the tactic will be considered detected.
  threshold: number;
  // Some tactics return additional fields:
  // * "vector_db":
  //   - "countOverMaxVectorScore" (number): The number of different vectors whose similarity score is above the 
  //       threshold.
  additionalFields: Record<string, any>;
}

export interface DetectResponse {
  // Whether prompt injection was detected.
  injectionDetected: boolean;
  // The result for each tactic that was executed.
  tacticResults: TacticResult[];
}

export class RebuffError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RebuffError";
  }
}

export class LeakLoggingError extends RebuffError {
  constructor(message: string, readonly leakId?: string) {
    super(message);
    this.name = "LeakLoggingError";
  }
}

export class VectorStoreInitializationError extends RebuffError {
  constructor(message: string) {
    super(message);
    this.name = "VectorStoreInitializationError";
  }
}

// The single, definitive outcome of a canary leak check. A failed log never
// resolves to `{ leaked: true, logged: false }`; it rejects instead, so a
// detected leak with an unknown alert state can never be mistaken for handled.
export interface CanaryLeakResult {
  // True only when the canary word was found in the completion.
  leaked: boolean;
  // True only when the leak was durably recorded (or a retry was deduplicated).
  logged: boolean;
  // Idempotency key reused on retry so the event is recorded exactly once and
  // stays pinned to the account that first accepted it.
  leakId: string;
}

export interface Rebuff {
  detectInjection(request: DetectRequest): Promise<DetectResponse>;

  addCanaryWord(
    prompt: string,
    canaryWord?: string,
    canaryFormat?: string
  ): [string, string];

  isCanaryWordLeaked(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome?: boolean,
    leakId?: string
  ): Promise<boolean>;

  checkCanaryLeak(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome?: boolean,
    leakId?: string
  ): Promise<CanaryLeakResult>;

  logLeakage(
    input: string,
    metaData: Record<string, string>,
    leakId?: string
  ): Promise<void>;
}
