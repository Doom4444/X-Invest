import re
import docx
import pandas as pd
import dateparser

from pypdf import PdfReader


# -----------------------------------
# DOCX Loader
# -----------------------------------
def load_docx(
    file_path: str
) -> str:

    text = ""

    try:

        document = docx.Document(
            file_path
        )

        for paragraph in document.paragraphs:

            paragraph_text = (
                paragraph.text.strip()
            )

            if paragraph_text:

                text += (
                    paragraph_text + "\n"
                )

    except Exception as e:

        print(

            f"Error reading DOCX "
            f"{file_path}: {e}"
        )

    return text


# -----------------------------------
# PDF Loader
# -----------------------------------
def load_pdf(
    file_path: str
) -> str:

    text = ""

    try:

        reader = PdfReader(
            file_path
        )

        for page in reader.pages:

            page_text = (
                page.extract_text()
            )

            if page_text:

                text += (
                    page_text.strip()
                    + "\n"
                )

    except Exception as e:

        print(

            f"Error reading PDF "
            f"{file_path}: {e}"
        )

    return text


# -----------------------------------
# TXT Loader
# -----------------------------------
def load_text(
    file_path: str
) -> str:

    try:

        with open(

            file_path,

            "r",

            encoding="utf-8"
        ) as f:

            return f.read()

    except Exception as e:

        print(

            f"Error reading TXT "
            f"{file_path}: {e}"
        )

        return ""


# -----------------------------------
# XLSX Loader
# -----------------------------------
def load_xlsx(
    file_path: str
) -> str:

    text = ""

    try:

        sheets = pd.read_excel(

            file_path,

            sheet_name=None
        )

        for sheet_name, df in sheets.items():

            # ----------------------------
            # Remove empty rows
            # ----------------------------
            df = df.dropna(
                how="all"
            )

            # ----------------------------
            # Skip empty sheets
            # ----------------------------
            if df.empty:

                continue

            text += (

                f"\nSheet: "

                f"{sheet_name}\n"
            )

            # ----------------------------
            # Dataframe to text
            # ----------------------------
            text += df.to_string(
                index=False
            )

            text += "\n"

    except Exception as e:

        print(

            f"Error reading XLSX "
            f"{file_path}: {e}"
        )

    return text


# -----------------------------------
# Main Loader
# -----------------------------------
def load_document(
    file_path: str
) -> str:

    file_path_lower = (
        file_path.lower()
    )

    # --------------------------------
    # PDF
    # --------------------------------
    if file_path_lower.endswith(
        ".pdf"
    ):

        return load_pdf(
            file_path
        )

    # --------------------------------
    # DOCX
    # --------------------------------
    elif file_path_lower.endswith(
        ".docx"
    ):

        return load_docx(
            file_path
        )

    # --------------------------------
    # TXT
    # --------------------------------
    elif file_path_lower.endswith(
        ".txt"
    ):

        return load_text(
            file_path
        )

    # --------------------------------
    # XLSX
    # --------------------------------
    elif file_path_lower.endswith(
        ".xlsx"
    ):

        return load_xlsx(
            file_path
        )

    # --------------------------------
    # Fallback
    # --------------------------------
    else:

        try:

            with open(

                file_path,

                "r",

                encoding="utf-8"
            ) as f:

                print(

                    "[Fallback] "

                    f"Trying to read "
                    f"as text: {file_path}"
                )

                return f.read()

        except Exception:

            print(

                "[Skipped] "

                f"Unsupported file "
                f"type: {file_path}"
            )

            return ""


# -----------------------------------
# Cleaning
# -----------------------------------
def clean_text(
    text: str
) -> str:

    """
    Clean text before chunking
    """

    if not text:

        return ""

    # --------------------------------
    # Remove null chars
    # --------------------------------
    text = text.replace(
        "\x00",
        " "
    )

    # --------------------------------
    # Normalize whitespace
    # --------------------------------
    text = re.sub(

        r"\s+",

        " ",

        text
    )

    # --------------------------------
    # Remove repeated punctuation
    # --------------------------------
    text = re.sub(

        r"\.{3,}",

        "...",

        text
    )

    # --------------------------------
    # Remove weird unicode chars
    # --------------------------------
    text = re.sub(

        r"[^\x00-\x7F]+",

        " ",

        text
    )

    return text.strip()


# -----------------------------------
# Smart Chunking with Overlap
# -----------------------------------
def chunk_text(

    text: str,

    chunk_size: int = 800,

    overlap: int = 100
):

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        # ----------------------------
        # Initial chunk end
        # ----------------------------
        end = start + chunk_size

        # ----------------------------
        # Clamp to text length
        # ----------------------------
        end = min(
            end,
            text_length
        )

        # ----------------------------
        # Try semantic boundary
        # ----------------------------
        if end < text_length:

            sentence_break = max(

                text.rfind(
                    ".",
                    start,
                    end
                ),

                text.rfind(
                    "\n",
                    start,
                    end
                )
            )

            # ------------------------
            # Use semantic boundary
            # ------------------------
            if (

                sentence_break != -1

                and sentence_break > start
            ):

                end = (
                    sentence_break + 1
                )

        # ----------------------------
        # Extract chunk
        # ----------------------------
        chunk = text[
            start:end
        ].strip()

        # ----------------------------
        # Skip tiny garbage chunks
        # ----------------------------
        if len(chunk) < 30:

            start = end

            continue

        # ----------------------------
        # Store chunk
        # ----------------------------
        chunks.append(chunk)

        # ----------------------------
        # Calculate next start
        # ----------------------------
        new_start = end - overlap

        # ----------------------------
        # Prevent infinite loop
        # ----------------------------
        if new_start <= start:

            new_start = end

        start = new_start

        # ----------------------------
        # Final safety
        # ----------------------------
        if start >= text_length:

            break

    return chunks


# -----------------------------------
# Date Extraction
# -----------------------------------
def extract_date(
    question: str
):

    """
    Extract date from user query
    """

    try:

        parsed_date = (
            dateparser.parse(

                question,

                settings={

                    # --------------------
                    # Forecast-oriented
                    # --------------------
                    "PREFER_DATES_FROM":
                        "future",

                    # --------------------
                    # Stable parsing
                    # --------------------
                    "RELATIVE_BASE":
                        pd.Timestamp.now()
                }
            )
        )

        return parsed_date

    except Exception as e:

        print(

            "[DateExtractor] "

            f"Error: {e}"
        )

        return None