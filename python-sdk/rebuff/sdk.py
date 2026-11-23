import secrets
from typing import Any, Dict, List, Optional, Tuple, Union

import rebuff.detect_pi_openai as _pi_detection_module
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from rebuff.detect_pi_heuristics import detect_prompt_injection_using_heuristic_on_input
from rebuff.detect_pi_openai import (
    PI_DETECTION_TEMPLATE_VERSION,
    call_openai_to_detect_pi,
    parse_score_from_completion,
    pi_detection_boundary,
    pi_detection_boundary_hits,
    render_prompt_for_pi_detection,
    user_input_digest,
)
from rebuff.detect_pi_vectorbase import detect_pi_using_vector_database, init_pinecone
from rebuff.errors import RebuffError


_ORIGINAL_RENDER_PROMPT = render_prompt_for_pi_detection
_ORIGINAL_CALL_OPENAI = call_openai_to_detect_pi


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
    # Evidence shared by the template, model call and score consumer.  These
    # fields are optional so existing callers that construct the model with
    # only the original fields keep working.
    template_version: Optional[str] = None
    raw_model_output: Optional[str] = None
    raw_model_response: Optional[str] = None
    raw_score: Optional[float] = None
    input_sha256: Optional[str] = None
    input_digest: Optional[str] = None
    input_length: Optional[int] = None
    boundary_id: Optional[str] = None
    boundary_hits: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)

    # Compatibility aliases for evidence consumers that use the model_*
    # spelling.  These are properties, not fields, so they do not create new
    # serialized response keys.
    @property
    def model_output(self) -> Optional[str]:
        return self.raw_model_output

    @property
    def model_raw_response(self) -> Optional[str]:
        return self.raw_model_response

    @property
    def model_error(self) -> Optional[str]:
        error = self.evidence.get("error")
        return str(error) if error is not None else None


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

    def initialize_pinecone(self) -> None:
        self.vector_store = init_pinecone(
            self.pinecone_apikey,
            self.pinecone_index,
            self.openai_apikey,
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

        if not isinstance(user_input, str):
            raise RebuffError("user_input must be a string")

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

        raw_completion: Any = None
        if check_llm:
            renderer = (
                render_prompt_for_pi_detection
                if render_prompt_for_pi_detection is not _ORIGINAL_RENDER_PROMPT
                else _pi_detection_module.render_prompt_for_pi_detection
            )
            rendered_input = renderer(user_input)
            caller = (
                call_openai_to_detect_pi
                if call_openai_to_detect_pi is not _ORIGINAL_CALL_OPENAI
                else _pi_detection_module.call_openai_to_detect_pi
            )
            try:
                model_response = caller(
                    rendered_input, self.openai_model, self.openai_apikey
                )
            except Exception as error:
                evidence = self._model_evidence(user_input, None, error=str(error))
                raise RebuffError(
                    "Language model detection failed; refusing to publish a safe score",
                    evidence=evidence,
                ) from error

            if not isinstance(model_response, dict):
                model_response = {"completion": model_response}

            raw_completion = model_response.get("completion")
            if raw_completion is None:
                for score_key in ("score", "raw_score"):
                    if score_key in model_response:
                        # Consumers that already parsed the raw response share
                        # the same score contract rather than falling back to
                        # 0.0.
                        raw_completion = model_response.get(score_key)
                        break
            model_error = model_response.get("error")
            if model_error:
                evidence = self._model_evidence(
                    user_input, raw_completion, error=str(model_error)
                )
                raise RebuffError(
                    "Language model detection failed; refusing to publish a safe score",
                    evidence=evidence,
                )

            try:
                # The raw model value is the only score source.  Missing,
                # empty, multi-value or otherwise malformed responses fail
                # closed instead of becoming a default 0.0.
                rebuff_model_score = parse_score_from_completion(raw_completion)
            except (TypeError, ValueError) as error:
                evidence = self._model_evidence(
                    user_input, raw_completion, error=str(error)
                )
                raise RebuffError(
                    "Language model returned an invalid score; refusing to publish a safe score",
                    evidence=evidence,
                ) from error

        else:
            rebuff_model_score = 0

        if (
            rebuff_heuristic_score > max_heuristic_score
            or rebuff_model_score > max_model_score
            or rebuff_vector_score > max_vector_score
        ):
            injection_detected = True

        model_evidence = (
            self._model_evidence(user_input, raw_completion)
            if check_llm
            else {}
        )
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
            template_version=PI_DETECTION_TEMPLATE_VERSION if check_llm else None,
            raw_model_output=(
                raw_completion if check_llm and isinstance(raw_completion, str) else None
            ),
            raw_model_response=(
                raw_completion if check_llm and isinstance(raw_completion, str) else None
            ),
            raw_score=rebuff_model_score if check_llm else None,
            input_sha256=user_input_digest(user_input),
            input_digest=user_input_digest(user_input),
            input_length=len(user_input),
            boundary_id=(
                pi_detection_boundary(user_input)[0] if check_llm else None
            ),
            boundary_hits=model_evidence.get("boundary_hits", []),
            evidence=model_evidence,
        )
        return rebuff_response

    def _model_evidence(
        self,
        user_input: str,
        raw_completion: Any,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build evidence that points back to the exact input and template."""

        token = pi_detection_boundary(user_input)[0]
        boundary_hits = pi_detection_boundary_hits(user_input)

        evidence: Dict[str, Any] = {
            "input_sha256": user_input_digest(user_input),
            "input_digest": user_input_digest(user_input),
            "input_hash": user_input_digest(user_input),
            "input_length": len(user_input),
            "original_input_length": len(user_input),
            "template_version": PI_DETECTION_TEMPLATE_VERSION,
            "boundary_id": token,
            "boundary_token": token,
            "model": self.openai_model,
            "model_output": raw_completion,
            "raw_model_output": raw_completion,
            "raw_response": raw_completion,
            "model_response": raw_completion,
            "model_raw_response": raw_completion,
            "completion": raw_completion,
            "boundary_hits": boundary_hits,
        }
        try:
            evidence["raw_score"] = parse_score_from_completion(raw_completion)
        except (TypeError, ValueError):
            evidence["raw_score"] = raw_completion
        evidence["parsed_score"] = evidence["raw_score"] if isinstance(
            evidence["raw_score"], float
        ) else None
        if error is not None:
            evidence["error"] = error
            evidence["model_error"] = error
            evidence["error_detail"] = error
        return evidence

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
    ) -> bool:
        """
        Checks if the canary word is leaked in the completion.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The canary word to check for leakage.
            log_outcome (bool, optional): Whether to log the outcome of the leakage check. Defaults to True.

        Returns:
            bool: True if the canary word is leaked, False otherwise.
        """
        if canary_word in completion:
            if log_outcome:
                self.log_leakage(user_input, completion, canary_word)
            return True
        return False

    def log_leakage(self, user_input: str, completion: str, canary_word: str) -> None:
        """
        Logs the leakage of a canary word.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The leaked canary word.
        """

        if self.vector_store is None:
            self.initialize_pinecone()

        self.vector_store.add_texts(
            [user_input],
            metadatas=[{"completion": completion, "canary_word": canary_word}],
        )

        return None
