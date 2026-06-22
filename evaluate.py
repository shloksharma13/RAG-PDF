import json

from retrieval import retrieve_context
from embeddings import get_embedding_model
from vector_store import load_vector_store
from sklearn.metrics import confusion_matrix

INDEX_PATH = "faiss_index"

embedding_model = get_embedding_model()

vector_store = load_vector_store(
    embedding_model,
    INDEX_PATH
)

with open("ground_truth.json") as f:
    dataset = json.load(f)

def precision_at_k(retrieved, relevant):

    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(retrieved & relevant) / len(retrieved)

def recall_at_k(retrieved, relevant):

    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(retrieved & relevant) / len(relevant)

def hit_rate(retrieved, relevant):

    return int(len(set(retrieved) & set(relevant)) > 0)

def reciprocal_rank(retrieved, relevant):

    for i, page in enumerate(retrieved):

        if page in relevant:
            return 1/(i+1)

    return 0

precisions = []
recalls = []
hits = []
mrrs = []

y_true = []
y_pred = []

for sample in dataset:

    results = retrieve_context(
        sample["question"],
        vector_store,
        k=3
    )

    retrieved_pages = [
        doc.metadata["page"]+1
        for doc, score in results
    ]

    relevant = sample["relevant_pages"]

    precisions.append(
        precision_at_k(retrieved_pages, relevant)
    )

    recalls.append(
        recall_at_k(retrieved_pages, relevant)
    )

    hits.append(
        hit_rate(retrieved_pages, relevant)
    )

    mrrs.append(
        reciprocal_rank(retrieved_pages, relevant)
    )

    actual = 1
    prediction = hit_rate(retrieved_pages, relevant)

    y_true.append(actual)
    y_pred.append(prediction)

cm = confusion_matrix(
    y_true,
    y_pred
)

print()

print("Precision@3 : ", sum(precisions)/len(precisions))

print("Recall@3 : ", sum(recalls)/len(recalls))

print("Hit Rate :", sum(hits)/len(hits))

print("MRR :", sum(mrrs)/len(mrrs))

print(cm)

