import {
  DetectRequest,
  DetectResponse,
  Rebuff,
  RebuffError,
} from "./interface";
import fetch from "node-fetch";
import { ApiConfig } from "./config";
import {
  DEFAULT_CANARY_FORMAT,
  embedCanary,
  isCanaryLeaked,
} from "./lib/canary";

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
    canaryWord?: string,
    canaryFormat = DEFAULT_CANARY_FORMAT
  ): [string, string] {
    const canary = embedCanary(prompt, canaryWord, canaryFormat);
    return [canary.prompt, canary.canaryWord];
  }

  isCanaryWordLeaked(
    userInput: string,
    completion: string,
    canaryWord: string,
    logOutcome = true,
    canaryFormat?: string
  ): boolean {
    // Check the same identity that was embedded, not an arbitrary substring.
    if (isCanaryLeaked(completion, canaryWord, canaryFormat)) {
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
    const canaryWord =
      metaData.canaryWord ?? metaData["canary_word"] ?? "";
    const data = {
      user_input: userInput,
      completion: metaData.completion ?? "",
      canaryWord,
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
