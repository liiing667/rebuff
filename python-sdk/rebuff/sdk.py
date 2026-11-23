from typing import Optional, Tuple, Union, cast

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from rebuff.canary import (
    CANARY_WORD_PLACEHOLDER,
    DEFAULT_CANARY_FORMAT,
    embed_canary,
    generate_canary_word,
    is_canary_leaked,
)
from rebuff.detect_pi_heuristics import detect_prompt_injection_using_heuristic_on_input
from rebuff.detect_pi_openai import (
    call_openai_to_detect_pi,
    render_prompt_for_pi_detection,
)
from rebuff.detect_pi_vectorbase import detect_pi_using_vector_database, init_pinecone


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
        return generate_canary_word(length)

    def add_canary_word(
        self,
        prompt: Union[str, PromptTemplate],
        canary_word: Optional[str] = None,
        canary_format: str = DEFAULT_CANARY_FORMAT,
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

        if isinstance(prompt, str):
            prompt_with_canary, canary = embed_canary(
                prompt, canary_word, canary_format, self.generate_canary_word
            )
            return prompt_with_canary, canary.canary_word

        elif isinstance(prompt, PromptTemplate):
            original_template = prompt.template
            prompt_with_canary, canary = embed_canary(
                original_template,
                canary_word,
                canary_format,
                self.generate_canary_word,
            )
            # Copy instead of mutating so concurrent calls cannot cross their
            # markers on a shared PromptTemplate instance.
            result = _copy_prompt_template(prompt)
            result.template = prompt_with_canary
            if CANARY_WORD_PLACEHOLDER in original_template:
                _drop_canary_input_variable(result)
            return result, canary.canary_word

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

        if self.vector_store is None:
            self.initialize_pinecone()

        self.vector_store.add_texts(
            [user_input],
            metadatas=[{"completion": completion, "canary_word": canary_word}],
        )

        return None


def _copy_prompt_template(prompt: PromptTemplate) -> PromptTemplate:
    """Copy a LangChain prompt template without relying on one pydantic version."""

    model_copy = getattr(prompt, "model_copy", None)
    if callable(model_copy):
        try:
            return cast(PromptTemplate, model_copy(deep=False))
        except TypeError:
            pass

    copy = getattr(prompt, "copy", None)
    if callable(copy):
        try:
            return cast(PromptTemplate, copy(deep=False))
        except TypeError:
            return cast(PromptTemplate, copy())

    return prompt


def _drop_canary_input_variable(prompt: PromptTemplate) -> None:
    """A filled ``{canary_word}`` placeholder is no longer an input variable."""

    input_variables = getattr(prompt, "input_variables", None)
    if not isinstance(input_variables, list):
        return
    if "canary_word" in input_variables:
        prompt.input_variables = [
            variable for variable in input_variables if variable != "canary_word"
        ]
