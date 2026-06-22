import ollama


SYSTEM_PROMPT = """
You are an AI assistant answering questions with an Employee Handbook as a reference.

Rules:

1. Answer ONLY using the provided handbook context.
2. If the answer isn't in the context, say:
   "I couldn't find that information in the handbook."
3. Never invent information.
4. Be concise and accurate.
5. Never hallucinate or make up answers.
6. Mention the handbook page numbers you used.
"""

def build_context(documents):

    context = []

    for doc in documents:

        page = doc.metadata["page"] + 1

        context.append(
            f"""
Page {page}

{doc.page_content}
"""
        )

    return "\n\n".join(context)

def build_prompt(question, context):

    return f"""
Context:

{context}

Question:

{question}

Answer:
"""


def generate_answer(question, documents):

    context = build_context(documents)

    prompt = build_prompt(question, context)

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return {
        "answer": response["message"]["content"],
        "pages": sorted(
            {
                doc.metadata["page"] + 1
                for doc in documents
            }
        ),
    }