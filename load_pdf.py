from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages.")
        return documents

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return []

    except Exception as e:
        print(f"Random Error: {e}")
        return []