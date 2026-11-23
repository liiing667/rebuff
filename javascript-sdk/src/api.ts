import {
  DetectRequest,
  DetectResponse,
  Rebuff,
  RebuffError,
} from "./interface";
import fetch from "node-fetch";
import crypto from "crypto";
import { ApiConfig } from "./config";
import { validateBoolean, validateThreshold } from "./validation";

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
  }: DetectRequest): Promise<DetectResponse> {
    if (userInput === null) {
      throw new RebuffError("userInput is required");
    }

    // Reject NaN, Infinity, negative or out-of-range thresholds and non-boolean
    // flags before the request leaves the client. The 0 and 1 boundaries and the
    // default thresholds all remain valid.
    validateThreshold(maxHeuristicScore, "maxHeuristicScore");
    validateThreshold(maxVectorScore, "maxVectorScore");
    validateThreshold(maxModelScore, "maxModelScore");
    validateBoolean(runHeuristicCheck, "runHeuristicCheck");
    validateBoolean(runVectorCheck, "runVectorCheck");
    validateBoolean(runLanguageModelCheck, "runLanguageModelCheck");

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

    // The injection verdict is authoritative and computed server-side. The
    // client must never overwrite it based on its own thresholds, otherwise a
    // tampered/buggy client could turn a "detected" verdict into "safe".
    if (typeof (responseData as any).injectionDetected !== "boolean") {
      throw new RebuffError(
        "Server response is missing a valid injectionDetected verdict"
      );
    }
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
