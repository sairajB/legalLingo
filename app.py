import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# ---- Configure Gemini API ----
genai.configure(api_key="AIzaSyBVbrz_Ngz9CmzFa04vWTcuFf-AqiwKUFY")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

# ---- Cache Embedding Model ----
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ---- Function to Extract Text from PDF ----
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
    return text

# ---- Create Chunks for Faster Processing ----
def create_text_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---- Create FAISS Vector Store ----
def create_vector_store(chunks):
    vector_dim = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(vector_dim)
    vectors = embedding_model.encode(chunks, show_progress_bar=True)
    index.add(np.array(vectors, dtype=np.float32))
    return index, vectors, chunks

# ---- Find Most Relevant Chunks using Cosine Similarity ----
def find_top_chunks(query, index, chunks, top_k=5):
    query_vector = embedding_model.encode([query])
    _, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)

    top_chunks = [chunks[i] for i in indices[0]]
    return " ".join(top_chunks)  # Combine top chunks for better accuracy

# ---- Stream Gemini Response ----
def stream_gemini_response(prompt):
    response = model.generate_content(prompt, stream=True)
    return "".join(chunk.text for chunk in response)

# ---- Initialize Streamlit ----
st.title("📄 PDF Q&A Chatbot")

# ---- Chat History Storage ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores (question, answer) tuples

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "temp_uploaded_file.pdf"  # Temporary storage
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save uploaded file

    # ---- Extract and Process PDF Text ----
    text = extract_text_from_pdf(pdf_path)
    chunks = create_text_chunks(text)

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_vector_store(chunks)

    index, vectors, chunks = st.session_state.vector_store
    st.success("✅ PDF Processed Successfully!")

    # ---- Display Chat History (Oldest Chats at the Bottom) ----
    st.write("### 📜 Chat History")
    for q, a in st.session_state.chat_history:  # Oldest chats first
        with st.chat_message("user"):
            st.write(f"**Q:** {q}")
        with st.chat_message("assistant"):
            st.write(f"**A:** {a}")

    # ---- Query Input ----
    query = st.text_input("Ask a question about the PDF:")

    if st.button("Submit") and query:
        # ---- Get Top Chunks ----
        relevant_text = find_top_chunks(query, index, chunks, top_k=5)

        # ---- Add System Prompt for Accuracy ----
        prompt = f"""You are an AI assistant answering questions based only on the given document.
Do not generate information that is not found in the document.

**Context:** {relevant_text}

**Question:** {query}

**Answer:**"""

        # ---- Generate Response ----
        answer = stream_gemini_response(prompt)

        # ---- Store in Chat History (Latest at Bottom) ----
        st.session_state.chat_history.append((query, answer))

        # ---- Refresh Page to Clear Input ----
        st.rerun()  # Clears input & updates the chat history