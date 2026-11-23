import {
  CanaryLeakResult,
  DetectRequest,
  DetectResponse,
  LeakLoggingError,
  Rebuff,
  RebuffError,
} from "./interface";
import fetch from "node-fetch";
import type { Response as FetchResponse } from "node-fetch";
import crypto from "crypto";
import { ApiConfig } from "./config";

// Indirection over the HTTP transport so it can be substituted in tests.
// Production code always uses the real node-fetch binding.
export type FetchLike = (
  url: string,
  init: { method: string; body: string; headers: Record<string, string> }
) => Promise<FetchResponse>;

function encodeString(message: string): string {
  return Buffer.from(message, "utf-8").toString("hex");
}

export interface LogApiRequest {
  user_input: string;
  completion: string;
  canaryWord: string;
  // Idempotency key reused on retry.
  leakId?: string;
}

export interface LogApiSuccessResponse {
  success: boolean;
  leakId?: string;
  accountId?: string;
  duplicate?: boolean;
}
function generateCanaryWord(length = 8): string {
  // Generate a secure random hexadecimal canary word
  return crypto.randomBytes(length / 2).toString("hex");
}

// Stable idempotency key derived from the leak event content, so retries of
// the same event dedupe even when the caller did not supply an explicit id.
function deriveLeakId(
  userInput: string,
  completion: string,
  canaryWord: string
): string {
  return crypto
    .createHash("sha256")
    .update(`${canaryWord}\x1f${userInput}\x1f${completion}`)
    .digest("hex");
}

export default class RebuffApi implements Rebuff {
  private readonly apiToken: string;

  private readonly apiUrl: string;

  private readonly headers: Record<string, string>;

  private fetchImpl: FetchLike = fetch as unknown as FetchLike;

  // Visible for testing: swap the transport without mutating module bindings.
  setFetchImplementation(impl: FetchLike): void {
    this.fetchImpl = impl;
  }

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
  }: DetectRequest): Promise<DetectResponse> {
    if (userInput === null) {
      throw new RebuffError("userInput is required");
    }
    const requestData: DetectRequest = {
      userInput: "",
      userInputBase64: encodeString(userInput),
      runHeuristicCheck: runHeuristicCheck,
      runVectorCheck: runVectorCheck,
      runLanguageModelCheck: runLanguageModelCheck,
      maxVectorScore,
      maxModelScore,
      maxHeuristicScore,
    };

    const response = await fetch(`${this.apiUrl}/api/detect`, {
      method: "POST",
      body: JSON.stringify(requestData),
      headers: this.headers,
    });

    const responseData = (await response.json()) as DetectResponse;

    if (!response.ok) {
      throw new RebuffError((responseData as any)?.message);
    }
    responseData.injectionDetected =
      responseData.heuristicScore > maxHeuristicScore ||
      responseData.modelScore > maxModelScore ||
      responseData.vectorScore.topScore > maxVectorScore;
    return responseData;
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

  async isCanaryWordLeaked(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome = true,
    leakId?: string
  ): Promise<boolean> {
    // Check if the canary word appears in the completion
    if (completion.includes(canaryWord)) {
      if (logOutcome) {
        // Await so a logging/alerting failure rejects instead of being lost.
        await this.logLeakage(userInput, { completion, canaryWord }, leakId);
      }
      return true;
    }
    return false;
  }

  async checkCanaryLeak(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome = true,
    leakId: string = crypto.randomBytes(16).toString("hex")
  ): Promise<CanaryLeakResult> {
    if (!completion.includes(canaryWord)) {
      return { leaked: false, logged: false, leakId };
    }
    if (!logOutcome) {
      return { leaked: true, logged: false, leakId };
    }
    await this.logLeakage(userInput, { completion, canaryWord }, leakId);
    return { leaked: true, logged: true, leakId };
  }

  async logLeakage(
    userInput: string,
    metaData: Record<string, string>,
    leakId?: string
  ): Promise<void> {
    const completion = metaData.completion ?? "";
    const canaryWord = metaData.canaryWord ?? metaData.canary_word ?? "";
    const eventId =
      leakId ?? deriveLeakId(userInput, completion, canaryWord);
    const data = {
      user_input: userInput,
      completion,
      canaryWord,
      leakId: eventId,
    };

    let response: FetchResponse;
    try {
      response = await this.fetchImpl(`${this.apiUrl}/api/log`, {
        method: "POST",
        body: JSON.stringify(data),
        headers: this.headers,
      });
    } catch (error) {
      // Fail closed: a detected leak with an unconfirmed log must never look
      // handled. Retry with the same leakId.
      throw new LeakLoggingError(
        `Canary leak detected but failed to log it (leak_id=${eventId}). ` +
          `Retry with the same leak id. Underlying error: ${String(error)}`,
        eventId
      );
    }

    if (!response.ok) {
      throw new LeakLoggingError(
        `Canary leak detected but the server rejected the log ` +
          `(leak_id=${eventId}, status=${response.status}). ` +
          `Retry with the same leak id.`,
        eventId
      );
    }
  }
}
