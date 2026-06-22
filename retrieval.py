from typing import List, Tuple
from langchain_core.documents import Document


def retrieve_context(
    question: str,
    vector_store,
    k: int = 5,
) -> List[Tuple[Document, float]]:

    documents = vector_store.similarity_search_with_score(
    query=question,
    k=k,
    )

    return documents