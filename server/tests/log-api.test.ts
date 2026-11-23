/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, beforeEach } from "mocha";
import { expect } from "chai";
import * as path from "path";

// Redirect the handler's path-aliased and npm imports to in-memory mocks
// before the handler module is loaded. This keeps the unit test dependency
// free (no tsconfig-paths) and hermetic (no Next/Supabase runtime needed).
const mocksDir = path.join(__dirname, "_mocks");
const mockTargets: Record<string, string> = {
  "@/lib/supabase": path.join(mocksDir, "supabase"),
  "@/lib/detect-helpers": path.join(mocksDir, "detect-helpers"),
  "@/lib/rebuff": path.join(mocksDir, "rebuff"),
  cors: path.join(mocksDir, "cors"),
  // The handler only imports Next.js types; no runtime value is needed.
  next: path.join(mocksDir, "next"),
};

const Module = require("module");
const originalResolveFilename = Module._resolveFilename;
Module._resolveFilename = function (
  request: string,
  ...rest: unknown[]
): string {
  if (mockTargets[request]) {
    return mockTargets[request];
  }
  // Generic "@/..." -> server root resolution for any future import.
  if (request.startsWith("@/")) {
    return path.join(__dirname, "..", request.slice(2));
  }
  return originalResolveFilename.call(this, request, ...rest);
};

// The "@types" alias only contributes compile-time interfaces; stub it.
const originalLoad = Module._load;
Module._load = function (request: string, ...rest: unknown[]): unknown {
  if (request === "@types") return {};
  return originalLoad.call(this, request, ...rest);
};

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { leakRows } = require("./_mocks/supabase") as typeof import("./_mocks/supabase");
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { vectorAlertState, vectorAlertCalls } =
  require("./_mocks/rebuff") as typeof import("./_mocks/rebuff");
// eslint-disable-next-line @typescript-eslint/no-var-requires
const handler = require("../pages/api/log").default;

function makeReqRes(apiKey: string, body: any) {
  const req: any = {
    method: "POST",
    headers: { authorization: `Bearer ${apiKey}` },
    body,
  };
  const res: any = {
    statusCode: 200,
    payload: undefined as any,
    status(code: number) {
      this.statusCode = code;
      return this;
    },
    json(payload: any) {
      this.payload = payload;
      return this;
    },
  };
  return { req, res };
}

describe("POST /api/log idempotency and fail-closed behavior", () => {
  beforeEach(() => {
    leakRows.length = 0;
    vectorAlertCalls.length = 0;
    vectorAlertState.shouldFail = false;
  });

  it("records a leak once and pins it to the first account across retries", async () => {
    const body = {
      user_input: "ui",
      completion: "comp",
      canaryWord: "canary",
      leakId: "leak-X",
    };

    const first = makeReqRes("key-account-A", body);
    await handler(first.req, first.res);
    expect(first.res.statusCode).to.equal(200);
    expect(first.res.payload.success).to.equal(true);
    expect(first.res.payload.accountId).to.equal("account-A");
    expect(first.res.payload.duplicate).to.equal(false);

    // Retry with a DIFFERENT api key (rotated key -> different account lookup).
    const second = makeReqRes("key-account-B", body);
    await handler(second.req, second.res);
    expect(second.res.statusCode).to.equal(200);
    expect(second.res.payload.duplicate).to.equal(true);
    // The authoritative owner is still the original account, never the new one.
    expect(second.res.payload.accountId).to.equal("account-A");

    expect(leakRows).to.have.lengthOf(1);
    expect(leakRows[0].account_id).to.equal("account-A");
  });

  it("fails closed when the leak cannot be pinned to an account", async () => {
    const { req, res } = makeReqRes("master-key", {
      user_input: "ui",
      completion: "comp",
      canaryWord: "canary",
      leakId: "leak-nopin",
    });
    await handler(req, res);
    // Not 2xx: the client treats this as a failure and retries.
    expect(res.statusCode).to.be.greaterThan(299);
    expect(leakRows).to.have.lengthOf(0);
  });

  it("fails closed when vector alerting fails, and retry dedupes", async () => {
    const body = {
      user_input: "ui",
      completion: "comp",
      canaryWord: "canary",
      leakId: "leak-Y",
    };

    vectorAlertState.shouldFail = true;
    const failed = makeReqRes("key-account-A", body);
    await handler(failed.req, failed.res);
    expect(failed.res.statusCode).to.equal(500);

    // The durable row exists from the first attempt; retry must not duplicate
    // it and reports the authoritative (already-accepted) state.
    vectorAlertState.shouldFail = false;
    const retried = makeReqRes("key-account-B", body);
    await handler(retried.req, retried.res);
    expect(retried.res.statusCode).to.equal(200);
    expect(retried.res.payload.duplicate).to.equal(true);
    expect(leakRows).to.have.lengthOf(1);
  });

  it("rejects requests without a valid api key", async () => {
    const { req, res } = makeReqRes("bad-key", {
      user_input: "ui",
      completion: "comp",
      canaryWord: "canary",
    });
    await handler(req, res);
    expect(res.statusCode).to.equal(401);
    expect(leakRows).to.have.lengthOf(0);
  });
});
