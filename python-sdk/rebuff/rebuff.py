import secrets
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field


class TacticName(str, Enum):
    # A series of heuristics are used to determine whether the input is prompt injection.
    heuristic = "heuristic"
    # A language model is asked if the input appears to be prompt injection.
    language_model = "language_model"
    # A vector database of known prompt injection attacks is queried for similarity.
    vector_db = "vector_db"


class TacticOverride(BaseModel):
    # The name of the tactic to override.
    name: TacticName
    # The threshold to use for this tactic. If the score is above this threshold,
    # the tactic will be considered detected. If not specified, the default
    # threshold for the tactic will be used.
    threshold: Optional[float] = None
    # Whether to run this tactic. Defaults to true if not specified.
    run: Optional[bool] = None


class TacticResult(BaseModel):
    # The name of the tactic.
    name: str
    # The score for the tactic, between 0 and 1. The closer to 1, the more
    # likely that this is a prompt injection attempt.
    score: float
    # Whether this tactic evaluated the input as a prompt injection attempt.
    detected: bool
    # The threshold used for this tactic.
    threshold: float
    # Some tactics return additional fields, e.g. "countOverMaxVectorScore".
    additionalFields: Dict[str, Any] = Field(default_factory=dict)


class DetectApiRequest(BaseModel):
    userInput: str
    userInputBase64: Optional[str] = None
    tacticOverrides: List[TacticOverride] = Field(default_factory=list)


class DetectApiSuccessResponse(BaseModel):
    # Whether prompt injection was detected.
    injectionDetected: bool
    # The result for each tactic that was executed, i.e. the evidence of
    # which checks actually ran.
    tacticResults: List[TacticResult]


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
    ) -> DetectApiSuccessResponse:
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
            DetectApiSuccessResponse: The server's detection response, including which
                checks actually ran (tacticResults) and the final decision (injectionDetected).

        Raises:
            ValueError: If all detection checks are disabled.
        """
        if not (check_heuristic or check_vector or check_llm):
            # With every check switched off there is no evidence to base a
            # decision on, so refuse instead of reporting a "safe" result.
            raise ValueError(
                "At least one detection check must be enabled to detect prompt injection"
            )

        # Channel selection and thresholds form a single policy that the
        # server executes; the response carries the evidence of what ran.
        request_data = DetectApiRequest(
            userInput="",
            userInputBase64=encode_string(user_input),
            tacticOverrides=[
                TacticOverride(
                    name=TacticName.heuristic,
                    threshold=max_heuristic_score,
                    run=check_heuristic,
                ),
                TacticOverride(
                    name=TacticName.vector_db,
                    threshold=max_vector_score,
                    run=check_vector,
                ),
                TacticOverride(
                    name=TacticName.language_model,
                    threshold=max_model_score,
                    run=check_llm,
                ),
            ],
        )

        response = requests.post(
            f"{self.api_url}/api/detect",
            json=request_data.model_dump(),
            headers=self._headers,
        )

        response.raise_for_status()

        # The server response is the single source of truth: it declares which
        # checks actually ran (tacticResults) and the final decision
        # (injectionDetected). Do not re-evaluate scores locally.
        return DetectApiSuccessResponse.model_validate(response.json())

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
        data = {
            "user_input": user_input,
            "completion": completion,
            "canaryWord": canary_word,
        }
        response = requests.post(
            f"{self.api_url}/api/log", json=data, headers=self._headers
        )
        response.raise_for_status()
        return


def encode_string(message: str) -> str:
    return message.encode("utf-8").hex()
