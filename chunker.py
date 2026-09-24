from pdf_reader import extract_text


def split_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    pdf_path = "documents/detailed notes.pdf"

    text = extract_text(pdf_path)

    chunks = split_text(text)

    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")

    print(chunks[0])
