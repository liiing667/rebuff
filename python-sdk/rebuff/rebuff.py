from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from pydantic import BaseModel

from .canary import (
    CANARY_WORD_PLACEHOLDER,
    DEFAULT_CANARY_FORMAT,
    embed_canary,
    generate_canary_word,
    is_canary_leaked,
)


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
        return generate_canary_word(length)

    def add_canary_word(
        self,
        prompt: Any,
        canary_word: Optional[str] = None,
        canary_format: str = DEFAULT_CANARY_FORMAT,
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

        if isinstance(prompt, str):
            prompt_with_canary, canary = embed_canary(
                prompt, canary_word, canary_format, self.generate_canary_word
            )
            return prompt_with_canary, canary.canary_word

        prompt_template_classes = _get_prompt_template_classes()
        if prompt_template_classes and isinstance(prompt, prompt_template_classes):
            original_template = prompt.template
            prompt_with_canary, canary = embed_canary(
                original_template,
                canary_word,
                canary_format,
                self.generate_canary_word,
            )
            # Do not mutate the caller's template: concurrent calls sharing a
            # template must not stack each other's markers.
            result = _copy_prompt_template(prompt)
            result.template = prompt_with_canary
            if CANARY_WORD_PLACEHOLDER in original_template:
                _drop_canary_input_variable(result)
            return result, canary.canary_word

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
        canary_format: Optional[str] = None,
    ) -> bool:
        """
        Checks if the canary word is leaked in the completion.

        Args:
            user_input (str): The user input.
            completion (str): The completion generated by the AI.
            canary_word (str): The canary word to check for leakage.
            log_outcome (bool, optional): Whether to log the outcome of the leakage check. Defaults to True.
            canary_format (Optional[str], optional): The format used when the canary was embedded. When provided,
             the exact marker is checked as well. Defaults to None.

        Returns:
            bool: True if the canary word is leaked, False otherwise.
        """
        if is_canary_leaked(completion, canary_word, canary_format):
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


def _get_prompt_template_classes() -> Tuple[type, ...]:
    """Return the available LangChain PromptTemplate classes."""

    classes: List[type] = []
    try:
        from langchain_core.prompts import PromptTemplate as CorePromptTemplate

        classes.append(CorePromptTemplate)
    except ImportError:
        pass

    try:
        from langchain.prompts import PromptTemplate as LegacyPromptTemplate

        classes.append(LegacyPromptTemplate)
    except ImportError:
        pass

    try:
        import langchain

        legacy_prompt_template = getattr(langchain, "PromptTemplate", None)
        if legacy_prompt_template is not None:
            classes.append(legacy_prompt_template)
    except ImportError:
        pass

    # Deduplicate while preserving order.
    unique_classes: List[type] = []
    for prompt_template_class in classes:
        if prompt_template_class not in unique_classes:
            unique_classes.append(prompt_template_class)
    return tuple(unique_classes)


def _copy_prompt_template(prompt: Any) -> Any:
    """Copy a LangChain prompt template without relying on one pydantic version."""

    model_copy = getattr(prompt, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(deep=False)
        except TypeError:
            pass

    copy = getattr(prompt, "copy", None)
    if callable(copy):
        try:
            return copy(deep=False)
        except TypeError:
            return copy()

    # A template type without copy semantics: fall back to the old behaviour
    # rather than failing outright.
    return prompt


def _drop_canary_input_variable(prompt: Any) -> None:
    """A filled ``{canary_word}`` placeholder is no longer an input variable."""

    input_variables = getattr(prompt, "input_variables", None)
    if not isinstance(input_variables, list):
        return
    if "canary_word" in input_variables:
        prompt.input_variables = [
            variable for variable in input_variables if variable != "canary_word"
        ]
