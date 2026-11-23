import secrets
import hashlib
from typing import Any, Dict, Optional, Tuple, Union

import requests
from pydantic import BaseModel

from rebuff.errors import LeakLoggingError, RebuffError


def _derive_leak_id(
    user_input: str, completion: str, canary_word: str
) -> str:
    """
    Derives a stable idempotency key for a leak event from its content.

    Retrying the exact same leak (same canary word, user input and completion)
    therefore produces the same id even when the caller did not thread an
    explicit id through the retry, while callers that supply their own id keep
    full control.
    """
    digest = hashlib.sha256()
    digest.update(canary_word.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(user_input.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(completion.encode("utf-8"))
    return digest.hexdigest()


class CanaryLeakResult(BaseModel):
    """
    The single, definitive outcome of a canary leak check.

    Attributes:
        leaked (bool): True only when the canary word was found in the
            completion. If logging was requested and failed, an error is
            raised instead of returning ``leaked=True`` with an unknown
            alert state.
        logged (bool): True only when the server durably recorded the leak
            (or confirmed a previous recording of the same ``leak_id``).
        leak_id (str): Idempotency key identifying this specific leak
            event. Reusing it on retry guarantees the event is recorded
            exactly once, under the same account, regardless of which API
            key authenticates the retry.
    """

    leaked: bool
    logged: bool
    leak_id: str

    def __bool__(self) -> bool:
        # Preserve the ergonomic ``if rb.is_canary_word_leaked(...):`` usage.
        return self.leaked


class DetectApiRequest(BaseModel):
    userInput: str
    userInputBase64: Optional[str] = None
    runHeuristicCheck: bool
    runVectorCheck: bool
    runLanguageModelCheck: bool
    maxHeuristicScore: float
    maxModelScore: float
    maxVectorScore: float


class DetectApiSuccessResponse(BaseModel):
    heuristicScore: float
    modelScore: float
    vectorScore: Dict[str, float]
    runHeuristicCheck: bool
    runVectorCheck: bool
    runLanguageModelCheck: bool
    maxHeuristicScore: float
    maxModelScore: float
    maxVectorScore: float
    injectionDetected: bool


class ApiFailureResponse(BaseModel):
    error: str
    message: str


class Rebuff:
    def __init__(self, api_token: str, api_url: str = "https://playground.rebuff.ai"):
        self.api_token = api_token
        self.api_url = api_url
        self._headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def detect_injection(
        self,
        user_input: str,
        max_heuristic_score: float = 0.75,
        max_vector_score: float = 0.90,
        max_model_score: float = 0.9,
        check_heuristic: bool = True,
        check_vector: bool = True,
        check_llm: bool = True,
    ) -> Union[DetectApiSuccessResponse, ApiFailureResponse]:
        """
        Detects if the given user input contains an injection attempt.

        Args:
            user_input (str): The user input to be checked for injection.
            max_heuristic_score (float, optional): The maximum heuristic score allowed. Defaults to 0.75.
            max_vector_score (float, optional): The maximum vector score allowed. Defaults to 0.90.
            max_model_score (float, optional): The maximum model (LLM) score allowed. Defaults to 0.9.
            check_heuristic (bool, optional): Whether to run the heuristic check. Defaults to True.
            check_vector (bool, optional): Whether to run the vector check. Defaults to True.
            check_llm (bool, optional): Whether to run the language model check. Defaults to True.

        Returns:
            Tuple[Union[DetectApiSuccessResponse, ApiFailureResponse], bool]: A tuple containing the detection
                metrics and a boolean indicating if an injection was detected.
        """
        request_data = DetectApiRequest(
            userInput=user_input,
            userInputBase64=encode_string(user_input),
            runHeuristicCheck=check_heuristic,
            runVectorCheck=check_vector,
            runLanguageModelCheck=check_llm,
            maxVectorScore=max_vector_score,
            maxModelScore=max_model_score,
            maxHeuristicScore=max_heuristic_score,
        )

        response = requests.post(
            f"{self.api_url}/api/detect",
            json=request_data.dict(),
            headers=self._headers,
        )

        response.raise_for_status()

        response_json = response.json()
        success_response = DetectApiSuccessResponse.parse_obj(response_json)

        if (
            success_response.heuristicScore > max_heuristic_score
            or success_response.modelScore > max_model_score
            or success_response.vectorScore["topScore"] > max_vector_score
        ):
            # Injection detected
            success_response.injectionDetected = True
            return success_response
        else:
            # No injection detected
            success_response.injectionDetected = False
            return success_response

    @staticmethod
    def generate_canary_word(length: int = 8) -> str:
        """
        Generates a secure random hexadecimal canary word.

        Args:
            length (int, optional): The length of the canary word. Defaults to 8.

        Returns:
            str: The generated canary word.
        """
        return secrets.token_hex(length // 2)

    def add_canary_word(
        self,
        prompt: Any,
        canary_word: Optional[str] = None,
        canary_format: str = "<!-- {canary_word} -->",
    ) -> Tuple[Any, str]:
        """
        Adds a canary word to the given prompt which we will use to detect leakage.

        Args:
            prompt (Any): The prompt to add the canary word to.
            canary_word (Optional[str], optional): The canary word to add. If not provided, a random canary word will be
             generated. Defaults to None.
            canary_format (str, optional): The format in which the canary word should be added.
            Defaults to "<!-- {canary_word} -->".

        Returns:
            Tuple[Any, str]: A tuple containing the modified prompt with the canary word and the canary word itself.
        """

        # Generate a canary word if not provided
        if canary_word is None:
            canary_word = self.generate_canary_word()

        # Embed the canary word in the specified format
        canary_comment = canary_format.format(canary_word=canary_word)
        if isinstance(prompt, str):
            prompt_with_canary: str = canary_comment + "\n" + prompt
            return prompt_with_canary, canary_word

        try:
            import langchain

            if isinstance(prompt, langchain.PromptTemplate):
                prompt.template = canary_comment + "\n" + prompt.template
                return prompt, canary_word
        except ImportError:
            pass

        raise TypeError(
            f"prompt_template must be a PromptTemplate or a str, "
            f"but was {type(prompt)}"
        )

    def is_canary_word_leaked(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
        log_outcome: bool = True,
        leak_id: Optional[str] = None,
    ) -> bool:
        """
        Checks if the canary word is leaked in the completion.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The canary word to check for leakage.
            log_outcome (bool, optional): Whether to log the outcome of the leakage check. Defaults to True.
            leak_id (Optional[str], optional): Caller-supplied idempotency key for this leak
                event. Generated when not provided. The same value MUST be reused if a failed
                call is retried, so the event is recorded exactly once under the original
                account.

        Returns:
            bool: True if the canary word is leaked, False otherwise.

        Raises:
            LeakLoggingError: The canary word leaked but the leak could not be logged and
                alerted durably. The leak itself is still confirmed; retry with the same
                ``leak_id``. This never silently returns a "safe" verdict.
        """
        if canary_word in completion:
            if log_outcome:
                self.log_leakage(
                    user_input, completion, canary_word, leak_id=leak_id
                )
            return True
        return False

    def check_canary_leak(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
        log_outcome: bool = True,
        leak_id: Optional[str] = None,
    ) -> CanaryLeakResult:
        """
        Checks for a canary leak and returns its full, definitive state.

        Unlike :meth:`is_canary_word_leaked`, which only returns a boolean, this exposes
        whether the leak was durably logged and the ``leak_id`` used, so callers can retry
        a failed log deterministically and observe the final state.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The canary word to check for leakage.
            log_outcome (bool, optional): Whether to log the outcome of the leakage
                check. Defaults to True.
            leak_id (Optional[str], optional): Idempotency key reused on retry.

        Returns:
            CanaryLeakResult: The leak verdict together with the logging state and
                the idempotency key.

        Raises:
            LeakLoggingError: A leak was found but logging could not be confirmed.
        """
        event_id = leak_id or secrets.token_hex(16)
        if canary_word not in completion:
            return CanaryLeakResult(leaked=False, logged=False, leak_id=event_id)

        if not log_outcome:
            return CanaryLeakResult(leaked=True, logged=False, leak_id=event_id)

        self.log_leakage(
            user_input, completion, canary_word, leak_id=event_id
        )
        return CanaryLeakResult(leaked=True, logged=True, leak_id=event_id)

    def log_leakage(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
        leak_id: Optional[str] = None,
    ) -> None:
        """
        Logs the leakage of a canary word.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The leaked canary word.
            leak_id (Optional[str], optional): Idempotency key identifying this leak
                event. Generated when not provided. Reuse the same value when retrying
                after a failure so the server records the event once and pins it to the
                account of the first accepted attempt.

        Raises:
            LeakLoggingError: The server did not durably accept the leak log. The
                caller should retry with the same ``leak_id``.
        """
        # Deterministic when derived from content so retries of the same event
        # are idempotent even without an explicit caller-supplied id.
        event_id = leak_id or _derive_leak_id(
            user_input, completion, canary_word
        )
        data = {
            "user_input": user_input,
            "completion": completion,
            "canaryWord": canary_word,
            "leakId": event_id,
        }
        try:
            response = requests.post(
                f"{self.api_url}/api/log", json=data, headers=self._headers
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            # Fail closed: a detected leak whose alert/log cannot be confirmed must
            # never be reported as a clean, fully-handled event.
            raise LeakLoggingError(
                f"Canary leak detected but failed to log it "
                f"(leak_id={event_id}). Retry with the same leak_id. "
                f"Underlying error: {exc}",
                leak_id=event_id,
            ) from exc
        return


def encode_string(message: str) -> str:
    return message.encode("utf-8").hex()
