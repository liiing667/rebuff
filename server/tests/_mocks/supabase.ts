// In-memory fake of the Supabase admin client used by the /api/log handler.

export type StoredLeak = {
  id: number;
  account_id: string;
  user_input: string;
  completion: string | null;
  canary_word: string | null;
  leak_id: string | null;
};

export const leakRows: StoredLeak[] = [];
let nextId = 1;

function table() {
  return {
    select: (_cols: string) => {
      const chain: any = {
        _filterLeakId: null as string | null,
        eq(col: string, value: string) {
          if (col === "leak_id") chain._filterLeakId = value;
          return chain;
        },
        limit() {
          return chain;
        },
      };
      Object.defineProperty(chain, "then", {
        get() {
          return (resolve: (v: unknown) => void) => {
            const matched = chain._filterLeakId
              ? leakRows.filter((r) => r.leak_id === chain._filterLeakId)
              : leakRows;
            resolve({ data: matched, error: null });
          };
        },
      });
      return chain;
    },
    insert: (records: StoredLeak[]) =>
      new Promise((resolve) => {
        for (const rec of records) {
          if (rec.leak_id && leakRows.some((r) => r.leak_id === rec.leak_id)) {
            resolve({
              data: null,
              error: { code: "23505", message: "duplicate key value (23505)" },
            });
            return;
          }
          leakRows.push({ id: nextId++, ...rec });
        }
        resolve({ data: null, error: null });
      }),
  };
}

export const supabaseAdminClient = {
  from: (_name: string) => table(),
};
