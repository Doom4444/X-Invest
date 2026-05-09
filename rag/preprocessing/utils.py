import docx
from pypdf import PdfReader
import pandas as pd
import re


# -----------------------------
# DOCX Loader
# -----------------------------
def load_docx(file_path: str) -> str:
    text = ""

    try:
        document = docx.Document(file_path)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")

    return text


# -----------------------------
# PDF Loader
# -----------------------------
def load_pdf(file_path: str) -> str:
    text = ""

    try:
        reader = PdfReader(file_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text.strip() + "\n"

    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")

    return text


# -----------------------------
# TXT Loader
# -----------------------------
def load_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""


# -----------------------------
# XLSX Loader
# -----------------------------
def load_xlsx(file_path: str) -> str:
    text = ""

    try:
        sheets = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in sheets.items():
            text += f"\nSheet: {sheet_name}\n"
            text += df.to_string(index=False)
            text += "\n"

    except Exception as e:
        print(f"Error reading XLSX {file_path}: {e}")

    return text


# -----------------------------
# Main Loader
# -----------------------------
def load_document(file_path: str) -> str:

    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".pdf"):
        return load_pdf(file_path)

    elif file_path_lower.endswith(".docx"):
        return load_docx(file_path)

    elif file_path_lower.endswith(".txt"):
        return load_text(file_path)

    elif file_path_lower.endswith(".xlsx"):
        return load_xlsx(file_path)

    else:
        # fallback
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                print(f"[Fallback] Trying to read as text: {file_path}")
                return f.read()
        except Exception:
            print(f"[Skipped] Unsupported file type: {file_path}")
            return ""


# -----------------------------
# Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """
    Clean text before chunking
    """

    if not text:
        return ""

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    # remove weird characters
    text = text.replace("\x00", " ")

    return text.strip()


# -----------------------------
# Chunking with Overlap 🔥
# -----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):

    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # move with overlap
        start = end - overlap

    return chunks