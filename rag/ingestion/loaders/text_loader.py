def load_text(file_path: str) -> str:
    """
    Read plain text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""