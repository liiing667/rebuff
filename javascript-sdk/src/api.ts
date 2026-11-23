import {
  DetectResponse,
  Rebuff,
  RebuffError,
  TacticResult,
} from "./interface";
import fetch, { Response as FetchResponse } from "node-fetch";
import crypto from "crypto";
import { ApiConfig } from "./config";

function encodeString(message: string): string {
  return Buffer.from(message, "utf-8").toString("hex");
}

export interface LogApiRequest {
  user_input: string;
  completion: string;
  canaryWord: string;
}

export interface LogApiSuccessResponse {
  success: boolean;
}

// Request accepted by the remote Rebuff detection API. These are the
// long-standing ("old") fields the /api/detect endpoint understands.
export interface ApiDetectRequest {
  userInput: string;
  maxHeuristicScore?: number;
  maxVectorScore?: number;
  maxModelScore?: number;
  runHeuristicCheck?: boolean;
  runVectorCheck?: boolean;
  runLanguageModelCheck?: boolean;
}

// Response returned by the remote Rebuff detection API. The classic per-check
// fields (heuristicScore, modelScore, vectorScore, ...) remain available, and
// injectionDetected is always the caller-side decision computed from the same
// validated payload.
export interface ApiDetectResponse extends DetectResponse {
  heuristicScore: number;
  modelScore: number;
  vectorScore: Record<string, number>;
  runHeuristicCheck: boolean;
  runVectorCheck: boolean;
  runLanguageModelCheck: boolean;
  maxHeuristicScore: number;
  maxVectorScore: number;
  maxModelScore: number;
  injectionDetected: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function toRebuffError(err: unknown, context: string): RebuffError {
  const reason = err instanceof Error ? err.message : String(err);
  const wrapped = new RebuffError(`${context}: ${reason}`);
  // Keep the original failure (e.g. a fetch timeout) available to callers.
  (wrapped as { cause?: unknown }).cause = err;
  return wrapped;
}

function generateCanaryWord(length = 8): string {
  // Generate a secure random hexadecimal canary word
  return crypto.randomBytes(length / 2).toString("hex");
}

export default class RebuffApi implements Rebuff {
  private readonly apiToken: string;

  private readonly apiUrl: string;

  private readonly headers: Record<string, string>;

  constructor(config: ApiConfig) {
    this.apiToken = config.apiKey;
    this.apiUrl = config.apiUrl ?? "https://playground.rebuff.ai";
    this.headers = {
      Authorization: `Bearer ${this.apiToken}`,
      "Content-Type": "application/json",
    };
  }

  async detectInjection({
    userInput = "",
    maxHeuristicScore = 0.75,
    maxVectorScore = 0.9,
    maxModelScore = 0.9,
    runHeuristicCheck = true,
    runVectorCheck = true,
    runLanguageModelCheck = true,
  }: ApiDetectRequest): Promise<ApiDetectResponse> {
    if (userInput === null) {
      throw new RebuffError("userInput is required");
    }
    const requestData = {
      userInput: "",
      userInputBase64: encodeString(userInput),
      runHeuristicCheck: runHeuristicCheck,
      runVectorCheck: runVectorCheck,
      runLanguageModelCheck: runLanguageModelCheck,
      maxVectorScore,
      maxModelScore,
      maxHeuristicScore,
    };

    // A network failure or timeout must never turn into a "safe" verdict.
    let response: FetchResponse;
    try {
      response = await fetch(`${this.apiUrl}/api/detect`, {
        method: "POST",
        body: JSON.stringify(requestData),
        headers: this.headers,
      });
    } catch (err) {
      throw toRebuffError(err, "Rebuff detect request failed");
    }

    // Read the body exactly once. The same payload drives HTTP acceptance,
    // error mapping, schema validation and the caller's decision below, so a
    // retry always evaluates the response that actually arrived.
    let rawBody: string;
    try {
      rawBody = await response.text();
    } catch (err) {
      throw toRebuffError(err, "Rebuff detect response could not be read");
    }

    let payload: unknown = null;
    if (rawBody.trim() !== "") {
      try {
        payload = JSON.parse(rawBody);
      } catch {
        payload = null;
      }
    }

    // HTTP acceptance: any non-2xx status is an error, regardless of how
    // convincing the body looks.
    if (!response.ok) {
      const message =
        (isRecord(payload) &&
          ((typeof payload.message === "string" && payload.message) ||
            (typeof payload.error === "string" && payload.error))) ||
        `Rebuff API request failed with status ${response.status}`;
      throw new RebuffError(message);
    }

    // A 200 without a JSON object body cannot support any verdict.
    if (!isRecord(payload)) {
      throw new RebuffError(
        "Rebuff API returned an invalid response body: expected a JSON object"
      );
    }

    // Error mapping: an error body must not masquerade as a safe success,
    // even when it arrives with a 200 status.
    if (typeof payload.error === "string" && payload.error) {
      throw new RebuffError(
        typeof payload.message === "string" && payload.message
          ? payload.message
          : payload.error
      );
    }

    // Schema validation: a response missing the detection fields is
    // malformed, not "safe".
    const missingFields: string[] = [];
    if (!isFiniteNumber(payload.heuristicScore)) {
      missingFields.push("heuristicScore");
    }
    if (!isFiniteNumber(payload.modelScore)) {
      missingFields.push("modelScore");
    }
    if (
      !isRecord(payload.vectorScore) ||
      !isFiniteNumber(payload.vectorScore.topScore)
    ) {
      missingFields.push("vectorScore.topScore");
    }
    if (missingFields.length > 0) {
      throw new RebuffError(
        `Rebuff API response is missing required field(s): ${missingFields.join(
          ", "
        )}`
      );
    }

    // Caller decision, computed from the very same validated payload.
    const heuristicScore = payload.heuristicScore as number;
    const modelScore = payload.modelScore as number;
    const vectorTopScore = (payload.vectorScore as Record<string, number>)
      .topScore;
    const injectionDetected =
      heuristicScore > maxHeuristicScore ||
      modelScore > maxModelScore ||
      vectorTopScore > maxVectorScore;

    return {
      tacticResults: [] as TacticResult[],
      ...payload,
      injectionDetected,
    } as unknown as ApiDetectResponse;
  }

  addCanaryWord(
    prompt: string,
    canaryWord: string = generateCanaryWord(),
    canaryFormat = "<!-- {canary_word} -->"
  ): [string, string] {
    // Embed the canary word in the specified format
    const canaryComment = canaryFormat.replace("{canary_word}", canaryWord);
    const promptWithCanary = `${canaryComment}\n${prompt}`;
    return [promptWithCanary, canaryWord];
  }

  isCanaryWordLeaked(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome = true
  ): boolean {
    // Check if the canary word appears in the completion
    if (completion.includes(canaryWord)) {
      if (logOutcome) {
        this.logLeakage(userInput, { completion, canaryWord });
      }
      return true;
    }
    return false;
  }

  async logLeakage(
    userInput: string,
    metaData: Record<string, string>
  ): Promise<void> {
    const data = {
      userInput,
      metaData,
    };

    const response = await fetch(`${this.apiUrl}/api/log`, {
      method: "POST",
      body: JSON.stringify(data),
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
  }
}
