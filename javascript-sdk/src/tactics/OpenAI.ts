import { RebuffError, TacticName } from "../interface";
import { OpenAIApi } from "openai";
import { renderPromptForPiDetection } from "../lib/prompts";
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
        messages: [{ role: "user", content: renderPromptForPiDetection(input) }],
      });
  
      if (completion.data.choices.length === 0) {
        throw new Error("completion.data.choices.length === 0");
      }
      if (completion.data.choices[0].message === undefined) {
        throw new Error("completion.data.choices[0].message is undefined");
      }

      const score = parseFloat(completion.data.choices[0].message.content || "");
      if (isNaN(score)) {
        // A score we cannot parse must not be silently treated as 0 (safe).
        throw new Error("language model returned a non-numeric score");
      }
      return { score };
    } catch (error) {
      console.error("Error in callOpenAiToDetectPI:", error);
      throw new RebuffError("Error in getting score for large language model");
    }
  }
}
