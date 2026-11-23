export const runMiddleware = async () => undefined;

export const checkApiKey = async (apiKey: string) => {
  if (apiKey === "master-key") {
    return { success: true, message: "master" };
  }
  if (apiKey === "bad-key") {
    return { success: false, message: "Invalid API key" };
  }
  // A different key maps to a different account to simulate key rotation
  // between the initial request and its retry.
  const accountId =
    apiKey === "key-account-A" ? "account-A" : `account-for-${apiKey}`;
  return { success: true, message: "ok", account_id: accountId };
};
