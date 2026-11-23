from typing import Any, Dict, Optional

import pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings


_GENERATION_METADATA_KEYS = ("query_generation", "generation")


def _resolve_query_generation(
    vector_store: Any, query_generation: Optional[int]
) -> int:
    """
    Resolves the active query generation for a vector database lookup.

    An explicit ``query_generation`` always wins. Otherwise the generation is
    read from the vector store itself (``vector_store.generation``, either an
    attribute or a zero-argument callable). Stores that do not track
    generations at all fall back to generation 0.
    """
    if query_generation is not None:
        return query_generation

    generation = getattr(vector_store, "generation", None)
    if callable(generation):
        generation = generation()

    if generation is None:
        return 0

    return generation


def _result_query_generation(document: Any) -> Optional[int]:
    """
    Reads the query generation a result document was produced in, if the
    document carries one in its metadata.
    """
    metadata = getattr(document, "metadata", None) or {}
    for key in _GENERATION_METADATA_KEYS:
        if key in metadata:
            return metadata[key]

    return None


# https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.pinecone.Pinecone.html
def detect_pi_using_vector_database(
    input: str,
    similarity_threshold: float,
    vector_store: Pinecone,
    query_generation: Optional[int] = None,
) -> Dict:
    """
    Detects Prompt Injection using similarity search with vector database.

    Args:
        input (str): user input to be checked for prompt injection
        similarity_threshold (float): The threshold for similarity between entries in vector database and the user input.
        vector_store (Pinecone): Vector database of prompt injections
        query_generation (Optional[int]): The query generation this lookup belongs to. Results stamped
                                        with a different (stale) generation - e.g. produced before the
                                        index was refreshed - are discarded. When omitted, the generation
                                        is resolved from the vector store itself.

    Returns:
        Dict (str, Union[float, int]): top_score (float) that contains the highest score wrt similarity between vector database and the user input.
                                        count_over_max_vector_score (int) holds the count for times the similarity score (between vector database and the user input)
                                        came out greater than or equal to the similarity_threshold.
                                        query_generation (int) the query generation the scores and the count were collected in.
    """

    top_k = 20
    generation = _resolve_query_generation(vector_store, query_generation)
    results = vector_store.similarity_search_with_score(input, top_k)

    top_score = 0.0
    count_over_max_vector_score = 0

    for document, score in results:
        # Discard stale evidence: results stamped with an older query
        # generation (e.g. collected before the index was refreshed) must not
        # leak into the current judgment.
        result_generation = _result_query_generation(document)
        if result_generation is not None and result_generation != generation:
            continue

        # Unknown scores stay in the score set as 0.0 instead of being
        # silently dropped.
        if score is None:
            score = 0.0

        # Tied scores are still top scores.
        if score >= top_score:
            top_score = score

        if score >= similarity_threshold:
            count_over_max_vector_score += 1

    vector_score = {
        "top_score": top_score,
        "count_over_max_vector_score": count_over_max_vector_score,
        "query_generation": generation,
    }

    return vector_score


def init_pinecone(api_key: str, index: str, openai_api_key: str) -> Pinecone:
    """
    Initializes connection with the Pinecone vector database using existing (rebuff) index.

    Args:
        api_key (str): Pinecone API key
        index (str): Pinecone index name
        openai_api_key (str): Open AI API key

    Returns:
        vector_store (Pinecone)

    """
    if not api_key:
        raise ValueError("Pinecone apikey definition missing")

    pc = pinecone.Pinecone(api_key=api_key)
    pc_index = pc.Index(index)

    openai_embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key, model="text-embedding-ada-002"
    )

    vector_store = Pinecone(pc_index, openai_embeddings, text_key="input")

    return vector_store
