# app.py (merged continuation) ------------------------
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# optional libs
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# LLM helper libs - import safely
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

st.set_page_config(page_title="StudyMate", page_icon="📘")
st.title("📘 StudyMate - AI Academic Assistant")
st.caption("Steps 3–5: Upload → Extract → Chunk → Embed → Search (plus LLM synthesizer)")

# ---------------- Existing functions: extraction, chunking, embedding, retrieval ----------------
def extract_text_from_pdfs(uploaded_files):
    results = []
    for uf in uploaded_files:
        file_bytes = uf.read()
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                full_text = []
                for page in doc:
                    full_text.append(page.get_text("text") or "")
                results.append({
                    "name": uf.name,
                    "pages": doc.page_count,
                    "text": "\n".join(full_text)
                })
        except Exception as e:
            results.append({
                "name": uf.name,
                "pages": 0,
                "text": f"[Error reading this PDF: {e}]"
            })
    return results

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks

def build_chunks_from_texts(pdf_texts, chunk_size=1000, overlap=200):
    all_chunks = []
    for pdf in pdf_texts:
        if pdf["text"].strip():
            pieces = chunk_text(pdf["text"], chunk_size, overlap)
            for idx, ch in enumerate(pieces):
                all_chunks.append({
                    "source": pdf["name"],
                    "chunk_id": idx,
                    "text": ch
                })
    return all_chunks

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index_and_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings
    else:
        return None, embeddings

def retrieve_top_k(query, model, index, embeddings, chunks, top_k=4):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

    if index is not None:
        D, I = index.search(q_emb, top_k)
        indices = I[0].tolist()
        scores = D[0].tolist()
    else:
        sims = np.dot(embeddings, q_emb[0])
        idx_sorted = np.argsort(-sims)[:top_k]
        indices = idx_sorted.tolist()
        scores = sims[idx_sorted].tolist()

    results = []
    for rank, idx in enumerate(indices):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append({
            "rank": rank + 1,
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "score": float(scores[rank])
        })
    return results

# ---------------- New: build context & LLM generators ----------------
def build_context_from_hits(hits, max_chars=3000):
    pieces = []
    cur_len = 0
    for h in hits:
        piece = f"Source: {h['source']}, chunk {h.get('chunk_id', '-')}\n{h['text'].strip()}\n\n"
        if cur_len + len(piece) > max_chars:
            break
        pieces.append(piece)
        cur_len += len(piece)
    return "\n".join(pieces)

def generate_answer_openai(question, hits, model_name=None, max_context_chars=3000, max_tokens=300):
    """
    Returns assistant text or None if OpenAI not configured / error.
    """
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    context = build_context_from_hits(hits, max_chars=max_context_chars)
    prompt_system = (
        "You are StudyMate, an assistant that answers questions ONLY using the provided context. "
        "If the answer is not present in the context, say: 'I don't know based on the provided documents.' "
        "Keep the answer short (3-6 sentences) and mention the sources (document names / chunk ids) if relevant."
    )
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
    ]
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return text
    except Exception:
        return None

def generate_answer_huggingface(question, hits, model_id=None, max_context_chars=3000, max_new_tokens=300):
    """
    Calls Hugging Face Inference API (text-generation). Returns text or None.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key or not REQUESTS_AVAILABLE:
        return None
    model_name = model_id or os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    context = build_context_from_hits(hits, max_chars=max_context_chars)
    prompt = (
        "You are StudyMate. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know. "
        "Keep the answer concise and include source references (document names / chunk ids) if relevant.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2}},
            timeout=60
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # many HF inference responses are [{'generated_text': "..."}]
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        # or some models return dict
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        # fallback: return stringified response if present
        if isinstance(data, list) and len(data) > 0:
            # combine candidate texts
            cand = []
            for item in data:
                if isinstance(item, dict) and "generated_text" in item:
                    cand.append(item["generated_text"])
            if cand:
                return " ".join(cand).strip()
        return None
    except Exception:
        return None

def synthesize_answer(question, hits, provider_pref="Auto"):
    """
    provider_pref: "Auto", "OpenAI", "HuggingFace", "Fallback"
    Returns (answer_text, provider_str)
    """
    if provider_pref == "OpenAI":
        ans = generate_answer_openai(question, hits)
        return (ans, "OpenAI") if ans else (None, "OpenAI")
    if provider_pref == "HuggingFace":
        ans = generate_answer_huggingface(question, hits)
        return (ans, "HuggingFace") if ans else (None, "HuggingFace")

    # Auto: try OpenAI first, then HF
    ans = generate_answer_openai(question, hits)
    if ans:
        return ans, "OpenAI"
    ans = generate_answer_huggingface(question, hits)
    if ans:
        return ans, "HuggingFace"

    # Fallback: assemble short excerpt
    simple = " ".join(h["text"].strip()[:600] for h in hits)  # short
    fallback = f"(LLM not configured) Most relevant excerpts:\n\n{simple}"
    return fallback, "fallback"

# ---------------- UI: upload / chunk / index (your existing UI re-used) ----------------
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")
    if st.button("📥 Read text from PDFs"):
        with st.spinner("Extracting text..."):
            texts = extract_text_from_pdfs(uploaded_files)
            st.session_state["pdf_texts"] = texts
        st.success("✅ Text extracted.")

if "pdf_texts" in st.session_state:
    st.subheader("📄 Extracted Text Preview")
    for i, item in enumerate(st.session_state["pdf_texts"], start=1):
        st.markdown(f"**{i}. {item['name']} — {item['pages']} page(s)**")
        preview = item["text"][:1500] if item["text"] else ""
        if not preview.strip():
            st.info("⚠ This PDF seems to have little/no selectable text. If it’s scanned, we’ll add OCR later.")
        else:
            st.text_area("Preview", value=preview + ("..." if len(item["text"]) > 1500 else ""), height=220, key=f"prev_{i}")

if "pdf_texts" in st.session_state:
    st.subheader("✂ Chunk Your Documents")
    chunk_size = st.number_input("Chunk size (characters)", min_value=500, max_value=2000, value=1000, step=100)
    overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=800, value=200, step=50)

    if st.button("🔹 Create Chunks"):
        with st.spinner("Splitting text into chunks..."):
            chunks = build_chunks_from_texts(st.session_state["pdf_texts"], chunk_size, overlap)
            st.session_state["chunks"] = chunks
        st.success(f"✅ Done! Created {len(st.session_state['chunks'])} chunks.")

if "chunks" in st.session_state:
    st.subheader("📦 Chunk Preview (first 5)")
    for i, ch in enumerate(st.session_state["chunks"][:5], start=1):
        st.markdown(f"**Chunk {i} — {ch['source']}**")
        st.text_area("Text", value=ch["text"], height=150, key=f"chunk_{i}")

    st.divider()
    st.subheader("🔎 Step 5: Build semantic index (embeddings + FAISS)")
    if FAISS_AVAILABLE:
        st.info("FAISS detected on this environment — fast indexing enabled.")
    else:
        st.warning("FAISS not available — using NumPy fallback (slower but will work).")

    if st.button("⚙️ Build Semantic Index"):
        with st.spinner("Encoding chunks and building index (this may take a while)..."):
            model = load_embedder()
            index, embeddings = build_index_and_embeddings(st.session_state["chunks"], model)
            st.session_state["faiss_index"] = index
            st.session_state["embeddings"] = embeddings
        st.success("✅ Semantic index ready.")

# ---------------- UI: ask & synthesize ----------------
if st.session_state.get("faiss_index") is not None or st.session_state.get("embeddings") is not None:
    st.subheader("❓ Ask a question about your uploaded documents")
    query = st.text_input("Enter your question:", key="query_input")

    if query:
        with st.spinner("Retrieving relevant passages..."):
            model = load_embedder()
            index = st.session_state.get("faiss_index")
            embeddings = st.session_state.get("embeddings")
            chunks = st.session_state.get("chunks", [])
            hits = retrieve_top_k(query, model, index, embeddings, chunks, top_k=5)

        st.subheader("Top matching passages")
        for h in hits:
            st.markdown(f"**Rank {h['rank']} — {h['source']} — chunk {h['chunk_id']} — score: {h['score']:.4f}**")
            st.write(h["text"][:800] + ("..." if len(h["text"]) > 800 else ""))

        with st.expander("Sources"):
            sources = {}
            for h in hits:
                sources.setdefault(h["source"], []).append(h["chunk_id"])
            for src, ids in sources.items():
                st.markdown(f"- **{src}** — chunks: {ids}")

        # provider selector & generate
        provider_choice = st.selectbox("Choose answer provider", ["Auto (OpenAI → Hugging Face → fallback)", "OpenAI", "Hugging Face", "Fallback (no LLM)"])
        if st.button("🔮 Generate Answer"):
            pref = "Auto"
            if provider_choice.startswith("OpenAI"):
                pref = "OpenAI"
            elif provider_choice.startswith("Hugging Face"):
                pref = "HuggingFace"
            elif provider_choice.startswith("Fallback"):
                pref = "Fallback"

            with st.spinner("Generating concise answer..."):
                answer, provider = synthesize_answer(query, hits, provider_pref=pref)
            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Generated by: {provider}")

# ---------------- end of file ------------------------
