export const vectorAlertCalls: Array<{ input: string; meta: unknown }> = [];

// Tests flip this to simulate the vector-store alert path failing.
export const vectorAlertState = { shouldFail: false };

export const rebuff = {
  logLeakage: async (input: string, meta: unknown) => {
    if (vectorAlertState.shouldFail) {
      throw new Error("vector store down");
    }
    vectorAlertCalls.push({ input, meta });
  },
};
