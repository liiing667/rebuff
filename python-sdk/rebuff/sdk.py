import hashlib
import secrets
from typing import Optional, Tuple, Union

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from rebuff.errors import (
    LeakLoggingError,
    VectorStoreInitializationError,
)
from rebuff.rebuff import CanaryLeakResult
from rebuff.detect_pi_heuristics import detect_prompt_injection_using_heuristic_on_input
from rebuff.detect_pi_openai import (
    call_openai_to_detect_pi,
    render_prompt_for_pi_detection,
)
from rebuff.detect_pi_vectorbase import detect_pi_using_vector_database, init_pinecone


def _derive_leak_id(
    user_input: str, completion: str, canary_word: str
) -> str:
    """Stable idempotency key derived from the leak event content."""
    digest = hashlib.sha256()
    digest.update(canary_word.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(user_input.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(completion.encode("utf-8"))
    return digest.hexdigest()


class RebuffDetectionResponse(BaseModel):
    heuristic_score: float
    openai_score: float
    vector_score: float
    run_heuristic_check: bool
    run_vector_check: bool
    run_language_model_check: bool
    max_heuristic_score: float
    max_model_score: float
    max_vector_score: float
    injection_detected: bool


class RebuffSdk:
    def __init__(
        self,
        openai_apikey: str,
        pinecone_apikey: str,
        pinecone_index: str,
        openai_model: str = "gpt-3.5-turbo",
    ) -> None:
        self.openai_model = openai_model
        self.openai_apikey = openai_apikey
        self.pinecone_apikey = pinecone_apikey
        self.pinecone_index = pinecone_index
        self.vector_store = None
        # Leak ids already durably written through this client. This makes a
        # retry after a transient failure idempotent within the process while
        # preserving fail-closed semantics for genuine write failures.
        self._logged_leak_ids: set = set()

    def initialize_pinecone(self) -> None:
        if self.vector_store is not None:
            return
        try:
            self.vector_store = init_pinecone(
                self.pinecone_apikey,
                self.pinecone_index,
                self.openai_apikey,
            )
        except Exception as exc:
            # Fail closed: an uninitialized vector store must not be mistaken
            # for a clean "no injection / no leak" result.
            raise VectorStoreInitializationError(
                "Failed to initialize the Pinecone vector store; vector-based "
                "detection cannot produce a trustworthy result."
            ) from exc
        if self.vector_store is None:
            raise VectorStoreInitializationError(
                "Pinecone vector store initialization returned no store; "
                "vector-based detection cannot produce a trustworthy result."
            )

    def detect_injection(
        self,
        user_input: str,
        max_heuristic_score: float = 0.75,
        max_vector_score: float = 0.90,
        max_model_score: float = 0.90,
        check_heuristic: bool = True,
        check_vector: bool = True,
        check_llm: bool = True,
    ) -> RebuffDetectionResponse:
        """
        Detects if the given user input contains an injection attempt.

        Args:
            user_input (str): The user input to be checked for injection.
            max_heuristic_score (float, optional): The maximum heuristic score allowed. Defaults to 0.75.
            max_vector_score (float, optional): The maximum vector score allowed. Defaults to 0.90.
            max_model_score (float, optional): The maximum model (LLM) score allowed. Defaults to 0.90.
            check_heuristic (bool, optional): Whether to run the heuristic check. Defaults to True.
            check_vector (bool, optional): Whether to run the vector check. Defaults to True.
            check_llm (bool, optional): Whether to run the language model check. Defaults to True.

        Returns:
            RebuffDetectionResponse
        """

        injection_detected = False

        if check_heuristic:
            rebuff_heuristic_score = detect_prompt_injection_using_heuristic_on_input(
                user_input
            )

        else:
            rebuff_heuristic_score = 0

        if check_vector:
            self.initialize_pinecone()

            vector_score = detect_pi_using_vector_database(
                user_input, max_vector_score, self.vector_store
            )
            rebuff_vector_score = vector_score["top_score"]

        else:
            rebuff_vector_score = 0

        if check_llm:
            rendered_input = render_prompt_for_pi_detection(user_input)
            model_response = call_openai_to_detect_pi(
                rendered_input, self.openai_model, self.openai_apikey
            )

            rebuff_model_score = float(model_response.get("completion", 0))

        else:
            rebuff_model_score = 0

        if (
            rebuff_heuristic_score > max_heuristic_score
            or rebuff_model_score > max_model_score
            or rebuff_vector_score > max_vector_score
        ):
            injection_detected = True

        rebuff_response = RebuffDetectionResponse(
            heuristic_score=rebuff_heuristic_score,
            openai_score=rebuff_model_score,
            vector_score=rebuff_vector_score,
            run_heuristic_check=check_heuristic,
            run_language_model_check=check_llm,
            run_vector_check=check_vector,
            max_heuristic_score=max_heuristic_score,
            max_model_score=max_model_score,
            max_vector_score=max_vector_score,
            injection_detected=injection_detected,
        )
        return rebuff_response

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
        prompt: Union[str, PromptTemplate],
        canary_word: Optional[str] = None,
        canary_format: str = "<!-- {canary_word} -->",
    ) -> Tuple[Union[str, PromptTemplate], str]:
        """
        Adds a canary word to the given prompt which we will use to detect leakage.

        Args:
            prompt (Union[str, PromptTemplate]): The prompt to add the canary word to.
            canary_word (Optional[str], optional): The canary word to add. If not provided, a random canary word will be generated. Defaults to None.
            canary_format (str, optional): The format in which the canary word should be added. Defaults to "<!-- {canary_word} -->".

        Returns:
            Tuple[Union[str, PromptTemplate], str]: A tuple containing the modified prompt with the canary word and the canary word itself.
        """

        # Generate a canary word if not provided
        if canary_word is None:
            canary_word = self.generate_canary_word()

        # Embed the canary word in the specified format
        canary_comment = canary_format.format(canary_word=canary_word)

        if isinstance(prompt, str):
            prompt_with_canary: str = canary_comment + "\n" + prompt
            return prompt_with_canary, canary_word

        elif isinstance(prompt, PromptTemplate):
            prompt.template = canary_comment + "\n" + prompt.template
            return prompt, canary_word

        else:
            raise TypeError(
                f"prompt must be a langchain_core.prompts.PromptTemplate or a str, "
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
            leak_id (Optional[str], optional): Idempotency key for this leak event, reused
                when retrying after a failure.

        Returns:
            bool: True if the canary word is leaked, False otherwise.

        Raises:
            LeakLoggingError: A leak was found but it could not be logged durably. The
                leak is still confirmed; retry with the same ``leak_id``.
            VectorStoreInitializationError: Logging was requested but the vector store
                could not be initialized.
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
        Checks for a canary leak and returns its full, definitive state, including the
        idempotency ``leak_id`` and whether the event was durably logged.

        Raises:
            LeakLoggingError: A leak was found but logging could not be confirmed.
            VectorStoreInitializationError: The vector store could not be initialized.
        """
        event_id = leak_id or secrets.token_hex(16)
        if canary_word not in completion:
            return CanaryLeakResult(leaked=False, logged=False, leak_id=event_id)

        if not log_outcome:
            return CanaryLeakResult(leaked=True, logged=False, leak_id=event_id)

        self.log_leakage(user_input, completion, canary_word, leak_id=event_id)
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
                event. Reuse it when retrying so the same event is recorded at most once.

        Raises:
            VectorStoreInitializationError: The vector store cannot be initialized.
            LeakLoggingError: The vector store rejected the write. The leak is still a
                confirmed fact; the caller must retry rather than treat the event as safe.
        """
        # Deterministic when derived from content so retries of the same event
        # are idempotent even without an explicit caller-supplied id.
        event_id = leak_id or _derive_leak_id(
            user_input, completion, canary_word
        )

        # Idempotent retry: a previous attempt already durably recorded this exact event.
        if event_id in self._logged_leak_ids:
            return None

        if self.vector_store is None:
            self.initialize_pinecone()

        # initialize_pinecone() fails closed rather than leaving this unset, but
        # narrow the type defensively so the write can never run on None.
        assert self.vector_store is not None
        try:
            self.vector_store.add_texts(
                [user_input],
                metadatas=[
                    {
                        "completion": completion,
                        "canary_word": canary_word,
                        "leak_id": event_id,
                    }
                ],
            )
        except Exception as exc:
            # Fail closed: the canary leaked but the alert/log write failed. Do not
            # let the caller believe the event was handled or that the system is safe.
            raise LeakLoggingError(
                f"Canary leak detected but failed to log it to the vector store "
                f"(leak_id={event_id}). Retry with the same leak_id. "
                f"Underlying error: {exc}",
                leak_id=event_id,
            ) from exc

        self._logged_leak_ids.add(event_id)

        return None
