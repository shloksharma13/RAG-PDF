import os

from generation import generate_answer
from load_pdf import load_pdf
from chunking import chunk_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store, load_vector_store, save_vector_store
from retrieval import retrieve_context

PDF_PATH = "handbook.pdf"
INDEX_PATH = "faiss_index"


def build_vector_store(embedding_model):

    print("Building vector store")

    documents = load_pdf(PDF_PATH)
    chunks = chunk_documents(documents)

    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    save_vector_store(
        vector_store,
        INDEX_PATH,
    )

    return vector_store


def main():

    embedding_model = get_embedding_model()

    if os.path.exists(INDEX_PATH):
        vector_store = load_vector_store(
            embedding_model,
            INDEX_PATH,
        )
    else:
        vector_store = build_vector_store(
            embedding_model
        )

    print("RAG pipeline is ready!")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue    

        retrieved_results = retrieve_context(
            question,
            vector_store,
        )

        retrieved_docs = [doc for doc, _ in retrieved_results]

        result = generate_answer(
            question,
            retrieved_docs,
        )

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(result["answer"])

        print("\nSources:")
        print(result["pages"])


if __name__ == "__main__":
    main()