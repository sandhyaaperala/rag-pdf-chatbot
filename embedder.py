from sentence_transformers import SentenceTransformer
from chunker import split_text
from pdf_reader import extract_text

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    embeddings = model.encode(chunks)

    data = []

    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "chunk": chunk,
            "embedding": embedding.tolist()
        })

    return data


if __name__ == "__main__":

    pdf_path = "documents/detailed notesp.pdf"

    text = extract_text(pdf_path)

    chunks = split_text(text)

    embedded_chunks = create_embeddings(chunks)

    print("Total Chunks:", len(embedded_chunks))

    print("\nFirst Chunk:\n")
    print(embedded_chunks[0]["chunk"])

    print("\nEmbedding Length:")
    print(len(embedded_chunks[0]["embedding"]))
