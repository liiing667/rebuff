import { NextApiRequest, NextApiResponse } from "next";
import Cors from "cors";
import { runMiddleware, checkApiKey } from "@/lib/detect-helpers";
import { supabaseAdminClient } from "@/lib/supabase";
import { rebuff } from "@/lib/rebuff";
import {
  ApiFailureResponse,
  LogApiRequest,
  LogApiSuccessResponse,
} from "@types";

const cors = Cors({
  methods: ["POST"],
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<any>
) {
  await runMiddleware(req, res, cors);
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "not_allowed",
      message: "Method not allowed",
    } as ApiFailureResponse);
  }
  try {
    // Extract the API key from the Authorization header
    const apiKey = req.headers.authorization?.split(" ")[1];

    // Assert that the API key is present
    if (!apiKey) {
      return res.status(401).json({
        error: "unauthorized",
        message: "Missing API key",
      } as ApiFailureResponse);
    }

    // Check if the API key is valid and reduce the account balance
    const {
      success,
      message,
      account_id: accountId,
    } = await checkApiKey(apiKey);

    if (!success) {
      return res.status(401).json({
        error: "unauthorized",
        message: message,
      } as ApiFailureResponse);
    }

    // TODO: Add support for logging while using master key
    // Check if account id is present
    if (accountId === null || accountId === undefined) {
      console.log("ignoring log leak request, no account_id found");
      return res.status(200).json({
        success: true,
      } as LogApiSuccessResponse);
    }

    const body = (req.body ?? {}) as LogApiRequest & {
      userInput?: string;
      canary_word?: string;
      metaData?: Record<string, string>;
    };
    // Accept both the current request shape and the legacy shape emitted by
    // older JS clients, so a canary is always logged with the identity that
    // was actually checked.
    const userInput =
      body["user_input"] ??
      body.userInput ??
      body.metaData?.["user_input"] ??
      "";
    const completion =
      body.completion ?? body.metaData?.completion ?? "";
    const canaryWord =
      body.canaryWord ??
      body["canary_word"] ??
      body.metaData?.canaryWord ??
      body.metaData?.["canary_word"] ??
      "";

    const { error } = await supabaseAdminClient.from("leak_logs").insert([
      {
        // eslint-disable-next-line camelcase
        account_id: accountId,
        // eslint-disable-next-line camelcase
        user_input: userInput,
        completion: completion,
        // eslint-disable-next-line camelcase
        canary_word: canaryWord,
      },
    ]);

    if (error) {
      console.error(error);
      return res.status(500).json({
        error: "server_error",
        message: "Internal server error",
      } as ApiFailureResponse);
    }

    await rebuff.logLeakage(userInput, {
      accountId,
      // eslint-disable-next-line camelcase
      canary_word: canaryWord,
    });

    return res.status(200).json({
      success: true,
    } as LogApiSuccessResponse);
  } catch (error) {
    console.error("Error in log API:");
    console.error(error);
    console.trace();
    return res.status(500).json({
      error: "server_error",
      message: "Internal server error",
    } as ApiFailureResponse);
  }
}
