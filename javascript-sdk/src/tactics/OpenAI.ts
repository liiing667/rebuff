import { RebuffError, TacticName } from "../interface";
import { OpenAIApi } from "openai";
import {
  PI_DETECTION_TEMPLATE_VERSION,
  buildPiDetectionMessages,
  parseScoreFromCompletion,
  piDetectionBoundary,
  piDetectionBoundaryHits,
} from "../lib/prompts";
import Tactic, { TacticExecution } from "./Tactic";

export default class OpenAI implements Tactic {
  name = TacticName.LanguageModel;
  private openai: OpenAIApi;
  private model: string;
  public defaultThreshold: number;

  constructor(threshold: number, model: string, openai: OpenAIApi) {
    this.defaultThreshold = threshold;
    this.model = model;
    this.openai = openai;
  }

  async execute(input: string): Promise<TacticExecution> {
    try {
      const completion = await this.openai.createChatCompletion({
        model: this.model,
        messages: buildPiDetectionMessages(input),
      });

      if (completion.data.choices.length === 0) {
        throw new Error("completion.data.choices.length === 0");
      }
      if (completion.data.choices[0].message === undefined) {
        throw new Error("completion.data.choices[0].message is undefined");
      }

      const score = parseScoreFromCompletion(
        completion.data.choices[0].message.content
      );
      const boundary = piDetectionBoundary(input);
      return {
        score,
        additionalFields: {
          raw_response: completion.data.choices[0].message.content,
          template_version: PI_DETECTION_TEMPLATE_VERSION,
          boundary_id: boundary.token,
          boundary_hits: piDetectionBoundaryHits(input),
        },
      };
    } catch (error) {
      console.error("Error in callOpenAiToDetectPI:", error);
      throw new RebuffError("Error in getting score for large language model");
    }
  }
}
