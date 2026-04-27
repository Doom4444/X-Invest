import pandas as pd


def load_xlsx(file_path: str) -> str:
    """
    Extract text from Excel file.
    """
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