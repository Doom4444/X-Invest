import docx


def load_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file.
    """
    text = ""

    try:
        document = docx.Document(file_path)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")

    return text