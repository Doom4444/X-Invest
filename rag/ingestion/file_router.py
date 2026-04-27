import os

from rag.ingestion.loaders.pdf_loader import load_pdf
from rag.ingestion.loaders.docx_loader import load_docx
from rag.ingestion.loaders.xlsx_loader import load_xlsx
from rag.ingestion.loaders.text_loader import load_text


def load_file(file_path: str) -> str:
    """
    Detect file type and call the correct loader.
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)

    elif ext == ".docx":
        return load_docx(file_path)

    elif ext == ".xlsx":
        return load_xlsx(file_path)

    elif ext == ".txt":
        return load_text(file_path)

    else:
        print(f"Unsupported file type: {ext}")
        return ""