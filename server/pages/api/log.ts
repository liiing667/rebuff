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

// PostgREST/Supabase unique-violation error code (PostgreSQL SQLSTATE 23505).
const UNIQUE_VIOLATION = "23505";

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
      // Fail closed: we cannot pin the leak to an account, so we must not claim
      // it was durably logged. Return an error status (not 2xx, which clients
      // treat as a confirmed write) so the caller retries instead of assuming
      // the alert succeeded.
      return res.status(503).json({
        error: "account_unavailable",
        success: false,
        message: "Leak cannot be pinned to an account",
      } as ApiFailureResponse & LogApiSuccessResponse);
    }

    const {
      user_input: userInput,
      completion,
      canaryWord,
      leakId,
    } = req.body as LogApiRequest;

    // Idempotency: when a leak id is supplied, a retry of the same event must
    // resolve to the same row and therefore the same account. Look up any
    // previously accepted attempt before authenticating/writing again.
    if (leakId) {
      const { data: existing, error: lookupError } = await supabaseAdminClient
        .from("leak_logs")
        .select("id, account_id")
        .eq("leak_id", leakId)
        .limit(1);

      if (lookupError) {
        console.error("Error looking up existing leak log:", lookupError);
        return res.status(500).json({
          error: "server_error",
          message: "Internal server error",
        } as ApiFailureResponse);
      }

      if (existing && existing.length > 0) {
        // The event was already durably recorded. Report its authoritative
        // state, including the owning account, even if this retry used a
        // different API key. The event is never duplicated or moved.
        return res.status(200).json({
          success: true,
          leakId,
          // eslint-disable-next-line camelcase
          accountId: existing[0].account_id,
          duplicate: true,
        } as LogApiSuccessResponse);
      }
    }

    // Run every side effect that must succeed before reporting success. If any
    // of them fails the request errors out (fail closed) and the caller retries
    // with the same leak id; the unique constraint makes that retry safe.
    try {
      const { error } = await supabaseAdminClient.from("leak_logs").insert([
        {
          // eslint-disable-next-line camelcase
          account_id: accountId,
          // eslint-disable-next-line camelcase
          user_input: userInput,
          completion: completion,
          // eslint-disable-next-line camelcase
          canary_word: canaryWord,
          // eslint-disable-next-line camelcase
          leak_id: leakId ?? null,
        },
      ]);

      if (error) {
        // Concurrent retries can race past the lookup and both attempt the
        // insert. The unique constraint makes this safe: treat a unique
        // violation as an already-accepted duplicate of the same leak id and
        // return the authoritative row instead of failing or double-writing.
        if (
          leakId &&
          (error.code === UNIQUE_VIOLATION ||
            // Some client versions surface the code inside the message.
            String(error.message ?? "").includes(UNIQUE_VIOLATION))
        ) {
          const { data: raced, error: racedLookupError } =
            await supabaseAdminClient
              .from("leak_logs")
              .select("id, account_id")
              .eq("leak_id", leakId)
              .limit(1);

          if (!racedLookupError && raced && raced.length > 0) {
            return res.status(200).json({
              success: true,
              leakId,
              // eslint-disable-next-line camelcase
              accountId: raced[0].account_id,
              duplicate: true,
            } as LogApiSuccessResponse);
          }
        }

        console.error(error);
        // Fail closed: a leak that could not be persisted is an error, never a
        // success. The caller retries with the same leak id.
        return res.status(500).json({
          error: "server_error",
          message: "Internal server error",
        } as ApiFailureResponse);
      }

      // Vector-store alerting must also succeed before we acknowledge the
      // event. On a failure the outer catch returns 500; a retry carrying the
      // same leak id is deduplicated above instead of creating a second row.
      await rebuff.logLeakage(userInput, { accountId });
    } catch (error) {
      console.error("Failed to durably record/alert leak:", error);
      return res.status(500).json({
        error: "server_error",
        message: "Internal server error",
      } as ApiFailureResponse);
    }

    return res.status(200).json({
      success: true,
      leakId: leakId ?? null,
      accountId,
      duplicate: false,
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
