import json

from embedder import create_embeddings
from chunker import split_text
from pdf_reader import extract_text


def save_embeddings(data, filename="vector_store.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    pdf_path = "documents/detailed notes.pdf"

    text = extract_text(pdf_path)

    chunks = split_text(text)

    embedded_chunks = create_embeddings(chunks)

    save_embeddings(embedded_chunks)

    print("Vector store created successfully!")

    print("Total Chunks:", len(embedded_chunks))
