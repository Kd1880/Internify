import PyPDF2  # Library for reading PDF files


def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF file page by page and extracts all text.
    Input  : pdf_path  → path of PDF file
    Output : complete text string extracted from all pages

    Notes:
    - This uses PyPDF2's text extraction; it works for digital PDFs.
    - For scanned PDFs (images), integrate OCR (e.g., Tesseract).
    """
    combined_text = ""

    # Open the PDF file in binary read mode ('rb')
    with open(pdf_path, "rb") as file_handle:
        pdf_reader = PyPDF2.PdfReader(file_handle)

        # Loop through every page in the PDF
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            combined_text += page_text

    return combined_text
