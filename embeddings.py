import torch
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )