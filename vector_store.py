import os

from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    print("Vector store created.")

    return vector_store


def save_vector_store(vector_store, path="faiss_index"):
    vector_store.save_local(path)
    print(f"Vector store saved to '{path}'.")


def load_vector_store(embedding_model, path="faiss_index"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No vector store found at '{path}'."
        )

    print("Loading existing FAISS vector store")

    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    print("Vector store loaded.")

    return vector_store