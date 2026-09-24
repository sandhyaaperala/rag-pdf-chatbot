import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

from retrieval import retrieve


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Page configuration
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="centered"
)


# Title
st.title("📄 RAG PDF Chatbot")

st.caption(
    "Ask questions about your PDF using semantic search and AI."
)


# Question input
question = st.text_input(
    "Ask a question about your document:"
)


if question:

    # Retrieve relevant chunks
    with st.spinner("🔍 Searching the document..."):

        results = retrieve(
            question,
            top_k=3
        )

    # Combine retrieved chunks
    context = "\n\n".join(
        chunk for score, chunk in results
    )

    # Gemini prompt
    prompt = f"""
You are a helpful assistant answering questions about a PDF document.

Use ONLY the information provided in the context below.

If the answer is not present in the context, say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer clearly and simply.
"""

    # Generate answer
    with st.spinner("🤖 Generating answer..."):

        try:

            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )

            st.subheader("🤖 Answer")

            st.write(response.output_text)

        except Exception as e:

            st.error(
                "Gemini is currently unavailable or the API quota "
                "has been reached."
            )

    # Sources
    with st.expander("📚 View Retrieved Sources"):

        for score, chunk in results:

            st.write(
                f"**Similarity Score:** {score:.4f}"
            )

            st.write(chunk)

            st.divider()
