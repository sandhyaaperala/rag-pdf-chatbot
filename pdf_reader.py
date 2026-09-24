from pypdf import PdfReader


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


if __name__ == "__main__":
    pdf_path = "documents/detailed notes.pdf"

    extracted_text = extract_text(pdf_path)

    print(extracted_text)
