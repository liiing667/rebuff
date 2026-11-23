export interface ApiFailureResponse {
  error: string;
  message: string;
}

export interface LogApiRequest {
  user_input: string;
  completion: string;
  canaryWord: string;
  // Client-generated idempotency key. The same value is reused when retrying a
  // failed log call so the event is recorded exactly once and stays pinned to
  // the account that first accepted it, even if the retry authenticates with a
  // different (e.g. rotated) API key.
  leakId?: string;
}

export interface LogApiSuccessResponse {
  success: boolean;
  // Echoes the idempotency key and the account the leak is durably recorded
  // under, so a retrying caller can observe the final, authoritative state.
  leakId?: string;
  accountId?: string;
  // True when this call was a retry of an already-recorded event.
  duplicate?: boolean;
}

export interface PromptResponse {
  canary_word: string;
  canary_word_leaked: boolean;
  breach: boolean;
  output: string;
  detection: DetectResponse;
}

export interface DetectResponse {
  heuristicScore: number;
  modelScore: number;
  vectorScore: Record<string, number>;
  runHeuristicCheck: boolean;
  runVectorCheck: boolean;
  runLanguageModelCheck: boolean;
  maxHeuristicScore: number;
  maxVectorScore: number;
  maxModelScore: number;
  injectionDetected: boolean;
}

interface PlaygroundStats {
  breaches: {
    total: number;
    user: number;
  };
  detections: number;
  requests: number;
}

export interface AppState {
  apikey: string;
  credits: number;
  stats: PlaygroundStats;
}
export interface AppStateCtx {
  appState: AppState;
  promptLoading: boolean;
  accountLoading: boolean;
  attempts: Attempt[];
  refreshAppState: () => Promise<void>;
  refreshApikey: () => Promise<void>;
  submitPrompt: (prompt: PromptRequest) => Promise<void>;
  setPromptLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

export interface PromptRequest {
  userInput: string;
  runHeuristicCheck: boolean;
  runVectorCheck: boolean;
  runLanguageModelCheck: boolean;
}

export interface Attempt extends PromptResponse {
  timestamp: Date;
  input: string;
  error: Error | null;
}
