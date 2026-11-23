import {
  CanaryLeakResult,
  DetectRequest,
  DetectResponse,
  LeakLoggingError,
  Rebuff,
  RebuffError,
  TacticResult,
  VectorStoreInitializationError,
} from "./interface";
import crypto from "crypto";
import { SdkConfig } from "./config";
import initVectorStore from "./lib/vectordb";
import getOpenAIInstance from "./lib/openai";
import { VectorStore } from "langchain/vectorstores/base";
import { Document } from "langchain/document";
import Strategy from "./lib/Strategy";
import Heuristic from "./tactics/Heuristic";
import OpenAI from "./tactics/OpenAI";
import Vector from "./tactics/Vector";

function generateCanaryWord(length = 8): string {
  // Generate a secure random hexadecimal canary word
  return crypto.randomBytes(length / 2).toString("hex");
}

// Stable idempotency key derived from the leak event content. Retrying the
// exact same leak yields the same id even when the caller did not thread one
// explicitly, so the event is never recorded twice.
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

export default class RebuffSdk implements Rebuff {
  private sdkConfig: SdkConfig;
  private vectorStore: VectorStore | undefined;
  private strategies: Record<string, Strategy> | undefined;
  private defaultStrategy: string;
  // Leak ids already durably written through this client, making in-process
  // retries idempotent while keeping genuine write failures fail-closed.
  private loggedLeakIds: Set<string> = new Set();

  /**
   * @deprecated Use `RebuffSdk.init` instead.
   */
  constructor(config: SdkConfig) {
    // We're keeping this constructor for backwards compatibility. In the future, we can make it private and
    // simplify this class quite a bit.

    this.sdkConfig = config;
    this.defaultStrategy = "standard";
  }

  public static async init(config: SdkConfig): Promise<RebuffSdk> {
    const sdk = new RebuffSdk(config);
    sdk.vectorStore = await initVectorStore(config);
    sdk.strategies = await sdk.getStrategies();
    return sdk;
  }

  private async getStrategies(): Promise<Record<string, Strategy>> {
    if (this.strategies) {
      return this.strategies;
    }
    const openai = {
      conn: getOpenAIInstance(this.sdkConfig.openai.apikey),
      model: this.sdkConfig.openai.model || "gpt-3.5-turbo",
    };
    const heuristicScoreThreshold = 0.75;
    const vectorScoreThreshold = 0.9;
    const openaiScoreThreshold = 0.9;
    this.strategies = {
      // For now, this is the only strategy.
      "standard": {
        tactics: [
          new Heuristic(heuristicScoreThreshold),
          new Vector(vectorScoreThreshold, await this.getVectorStore()),
          new OpenAI(openaiScoreThreshold, openai.model, openai.conn),
        ]
      },
    };
    return this.strategies;
  }

  async detectInjection({
    userInput = "",
    userInputBase64 = "",
    tacticOverrides = [],
  }: DetectRequest): Promise<DetectResponse> {
    if (userInputBase64) {
      // Create a buffer from the hexadecimal string
      const userInputBuffer = Buffer.from(userInputBase64, "hex");
      // Decode the buffer to a UTF-8 string
      userInput = userInputBuffer.toString("utf-8");
    }
    if (!userInput) {
      throw new RebuffError("userInput is required");
    }

    const strategies = await this.getStrategies();
    let injectionDetected = false;
    let tacticResults: TacticResult[] = [];
    for (const tactic of strategies[this.defaultStrategy].tactics) {
      const tacticOverride = tacticOverrides.find(t => t.name === tactic.name);
      if (tacticOverride && tacticOverride.run === false) {
        continue;
      }
      const threshold = tacticOverride?.threshold ?? tactic.defaultThreshold;
      const execution = await tactic.execute(userInput, threshold);
      const result = {
        name: tactic.name,
        score: execution.score,
        threshold,
        detected: execution.score > threshold,
        additionalFields: execution.additionalFields ?? {},
      } as TacticResult;
      if (result.detected) {
        injectionDetected = true;
      }
      tacticResults.push(result);
    }

    return {
      injectionDetected,
      tacticResults,
    } as DetectResponse;
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
        // Must be awaited: a detected leak whose log fails has to surface as
        // an error instead of silently returning "leaked and handled".
        await this.logLeakage(
          userInput,
          { completion, "canary_word": canaryWord },
          leakId
        );
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
    await this.logLeakage(
      userInput,
      { completion, "canary_word": canaryWord },
      leakId
    );
    return { leaked: true, logged: true, leakId };
  }

  async getVectorStore(): Promise<VectorStore> {
    if (this.vectorStore) {
      return this.vectorStore;
    }
    try {
      this.vectorStore = await initVectorStore(this.sdkConfig);
    } catch (error) {
      // Fail closed: an uninitialized vector store must never downgrade a
      // detection/leak decision into an implicit "clean" result.
      throw new VectorStoreInitializationError(
        `Failed to initialize vector store: ${error}`
      );
    }
    if (!this.vectorStore) {
      throw new VectorStoreInitializationError(
        "Vector store initialization produced no store"
      );
    }
    return this.vectorStore;
  }

  async logLeakage(
    input: string,
    metaData: Record<string, string>,
    leakId?: string
  ): Promise<void> {
    const eventId =
      leakId ??
      deriveLeakId(
        input,
        metaData.completion ?? "",
        metaData.canaryWord ?? metaData.canary_word ?? ""
      );

    // Idempotent in-process retry.
    if (this.loggedLeakIds.has(eventId)) {
      return;
    }

    try {
      await (await this.getVectorStore()).addDocuments([new Document({
        metadata: { ...metaData, leak_id: eventId },
        pageContent: input,
      })]);
    } catch (error) {
      // Preserve the specific fail-closed class for vector store init failures.
      if (error instanceof VectorStoreInitializationError) {
        throw error;
      }
      // Fail closed: the canary leaked but the alert/log write failed. The
      // caller must retry with the same leak id, never assume a safe state.
      throw new LeakLoggingError(
        `Canary leak detected but failed to log it (leak_id=${eventId}). ` +
          `Retry with the same leak id. Underlying error: ${String(error)}`,
        eventId
      );
    }
    this.loggedLeakIds.add(eventId);
  }
}
