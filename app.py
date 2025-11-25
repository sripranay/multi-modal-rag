# app.py — full updated app with uploader, processing, index building, QA and safe rerun fallback
import streamlit as st
import os
import subprocess
import pathlib
import time

import config

# Local sample PDF created earlier (sandbox path)
# LOCAL SANDBOX PATH: /mnt/data/sample_multimodal.pdf
# WINDOWS SAMPLE PATH (same file in project raw folder):
# C:\Users\HP\Downloads\multi-model_assignment\multi-model_assignment\data\raw\sample_multimodal.pdf

SAMPLE_PDF_PATH = os.path.join(config.RAW_DATA_DIR, "qatar_test_doc.pdf")
# also accept the sample_multimodal uploaded to data/raw if present:
SAMPLE_MULTIMODAL = os.path.join(config.RAW_DATA_DIR, "sample_multimodal.pdf")

st.set_page_config(page_title="RAG multi-model", layout="wide")
st.title("Multi-Modal RAG")
st.markdown("Ask questions about uploaded PDFs (text, tables, images supported).")

# ---- session state initialization ----
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: File status + actions
with st.sidebar:
    st.header("File Status")
    if st.session_state.loaded:
        st.success("Ready to use")
        try:
            chunks_len = len(st.session_state.vector_store.chunks)
            text_count = sum(1 for c in st.session_state.vector_store.chunks if c.get("type") == "text")
            table_count = sum(1 for c in st.session_state.vector_store.chunks if c.get("type") == "table")
            image_count = sum(1 for c in st.session_state.vector_store.chunks if c.get("type") == "image")
            st.markdown(f"**Chunks:** {chunks_len} — text: {text_count}, tables: {table_count}, images: {image_count}")
        except Exception:
            pass

        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            # safe rerun fallback
            try:
                st.experimental_rerun()
            except AttributeError:
                st.info("Chat cleared. Please refresh the page to see changes.")
    else:
        st.warning("No vector store loaded (preprocessing/index may be required).")
        st.markdown("---")
        st.subheader("Setup (one-time)")
        st.markdown(
            "If you have not processed the sample or uploaded a new PDF, run these steps:\n\n"
            "1. `python process_document.py`  (extract text/tables/images)\n\n"
            "2. `python create_embeddings.py` (create embeddings + FAISS index)\n\n"
            "Then restart the app with `streamlit run app.py`."
        )

# ---------------------- Upload / sample selection ----------------------
st.markdown("---")
st.subheader("Upload or choose a PDF")

# ensure raw dir exists
pathlib.Path(config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)

uploaded_file = st.file_uploader("Upload a PDF file (optional)", type=["pdf"])
selected_file = None
if uploaded_file is not None:
    saved_path = os.path.join(config.RAW_DATA_DIR, uploaded_file.name)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded and saved: {uploaded_file.name}")
    selected_file = saved_path

col1, col2 = st.columns(2)
with col1:
    if st.button("Use sample (Qatar report)"):
        if os.path.exists(SAMPLE_PDF_PATH):
            st.success("Selected sample: Qatar report")
            selected_file = SAMPLE_PDF_PATH
        else:
            st.error("Sample Qatar PDF not found in data/raw/")
with col2:
    if st.button("Use generated multimodal sample"):
        if os.path.exists(SAMPLE_MULTIMODAL):
            st.success("Selected sample: sample_multimodal.pdf")
            selected_file = SAMPLE_MULTIMODAL
        else:
            st.error("sample_multimodal.pdf not found in data/raw/")

if selected_file:
    st.markdown(f"**Selected file:** `{selected_file}`")

    # Run processing and embedding scripts
    if st.button("Process & Build Index (run extraction + embeddings)"):
        with st.spinner("Running processing scripts — this may take a few minutes..."):
            # Try to run process_document.py with --input, else fallback
            try:
                proc = subprocess.run(
                    ["python", "process_document.py", "--input", selected_file],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    # fallback: call without args (rely on config.PDF_PATH)
                    proc2 = subprocess.run(["python", "process_document.py"], check=False, capture_output=True, text=True)
                    if proc2.stdout:
                        st.text(proc2.stdout[:2000])
                    if proc2.stderr:
                        st.text(proc2.stderr[:2000])
                else:
                    if proc.stdout:
                        st.text(proc.stdout[:2000])
            except Exception as e:
                st.error(f"Error while running process_document.py: {e}")

            # Create embeddings
            try:
                proc_e = subprocess.run(["python", "create_embeddings.py"], check=True, capture_output=True, text=True)
                if proc_e.stdout:
                    st.text(proc_e.stdout[:2000])
                st.success("Processing and indexing complete.")
                # try to reload the app to pick up the new index
                try:
                    st.experimental_rerun()
                except AttributeError:
                    st.info("Processing complete — please refresh the browser to load the new index.")
            except subprocess.CalledProcessError as ex:
                st.error("create_embeddings.py failed. See terminal for details.")
                if ex.stdout:
                    st.text(ex.stdout[:2000])
                if ex.stderr:
                    st.text(ex.stderr[:2000])

# ---------------------- Loading vector store on app start ----------------------
if not st.session_state.loaded:
    try:
        # prefer using your VectorStore class if available
        try:
            from vector_store import VectorStore
            vs = VectorStore(model_name=config.EMBEDDING_MODEL)
            vs.load(config.VECTOR_STORE_PATH)
            st.session_state.vector_store = vs
            # try to instantiate LLMQA
            try:
                from llm_qa import LLMQA, SimpleQA
                st.session_state.qa_system = LLMQA(model_name=config.LLM_MODEL)
            except Exception:
                st.warning("LLMQA failed to load; using SimpleQA fallback.")
                try:
                    from llm_qa import SimpleQA
                    st.session_state.qa_system = SimpleQA()
                except Exception:
                    st.session_state.qa_system = None
            st.session_state.loaded = True
        except Exception as e_vs:
            # fallback: try to load prebuilt pickle list (faiss_index_chunks.pkl)
            pkl_path = os.path.join(config.VECTOR_STORE_DIR, "faiss_index_chunks.pkl")
            if os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, "rb") as f:
                    chunks = pickle.load(f)
                class SimpleStore:
                    def __init__(self, chunks):
                        self.chunks = chunks
                    def search(self, query, k=5):
                        q = query.lower()
                        scores = []
                        for i, c in enumerate(self.chunks):
                            content = c.get("content","").lower()
                            score = sum(content.count(w) for w in q.split())
                            if score>0:
                                scores.append((score, i))
                        scores.sort(reverse=True)
                        results = []
                        for rank, (_, idx) in enumerate(scores[:k], start=1):
                            results.append({"chunk": self.chunks[idx], "score": None})
                        return results
                st.session_state.vector_store = SimpleStore(chunks)
                st.session_state.qa_system = None
                st.session_state.loaded = True
            else:
                st.session_state.loaded = False
    except Exception as ex_load:
        st.error(f"Error while auto-loading vector store: {ex_load}")
        st.session_state.loaded = False

# ---------------------- Main QA interface ----------------------
st.markdown("---")
if st.session_state.loaded:
    st.subheader("Ask a question about the document")
    query = st.text_input("Enter your question here...", key="query_input")
    k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=5)

    if st.button("Ask") and query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Searching and generating answer..."):
            # 1) retrieve top-k chunks
            hits = []
            try:
                results = st.session_state.vector_store.search(query, k=k)
                # normalise results (support different return formats)
                for r in results:
                    chunk = None
                    score = None
                    if isinstance(r, dict):
                        chunk = r.get("chunk") or r.get("metadata") or r.get("doc") or r
                        score = r.get("score", None)
                    else:
                        chunk = r
                    content = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
                    source = ""
                    if isinstance(chunk, dict):
                        source = chunk.get("source") or chunk.get("page") or ""
                    hits.append({"content": content, "source": source, "score": score})
            except Exception as e_search:
                st.error(f"Search failed: {e_search}")
                hits = []

            # 2) show retrieved contexts
            st.markdown("**Retrieved context (top results):**")
            for i, h in enumerate(hits, start=1):
                st.markdown(f"**{i}. Source:** {h.get('source','unknown')} — score: {h.get('score')}")
                st.write(h.get("content","")[:1000])

            # 3) generate answer (OpenAI preferred)
            answer = ""
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    import openai
                    openai.api_key = os.environ["OPENAI_API_KEY"]
                    prompt = "Use the following context to answer the question. If insufficient, say 'Insufficient information'.\n\n"
                    for i, h in enumerate(hits, start=1):
                        prompt += f"Context {i}: {h.get('content','')}\n\n"
                    prompt += f"Question: {query}\nAnswer:"
                    resp = openai.ChatCompletion.create(
                        model=os.environ.get("OPENAI_MODEL","gpt-3.5-turbo"),
                        messages=[{"role":"user","content":prompt}],
                        temperature=0
                    )
                    answer = resp["choices"][0]["message"]["content"].strip()
                except Exception as e_open:
                    st.warning(f"OpenAI call failed: {e_open} — falling back to local HF model.")

            if not answer:
                # fallback to HF text2text pipeline
                try:
                    from transformers import pipeline
                    generator = pipeline("text2text-generation", model=os.environ.get("HF_MODEL", "google/flan-t5-small"), device=-1)
                    prompt = "Answer using the context below. If insufficient, reply 'Insufficient information'.\n\n"
                    for i,h in enumerate(hits, start=1):
                        prompt += f"Context {i}: {h.get('content','')}\n\n"
                    prompt += f"Question: {query}\nAnswer:"
                    gen = generator(prompt, max_length=256, do_sample=False)
                    answer = gen[0]["generated_text"]
                except Exception as e_hf:
                    st.error("No LLM available to generate answers. Set OPENAI_API_KEY or install transformers.")
                    answer = "No LLM available to generate answers."

            # 4) display answer and store chat history
            st.markdown("**Answer:**")
            st.write(answer)
            st.session_state.chat_history.append({"role":"assistant", "content": answer, "citations": hits})

    # display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat history")
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
                if msg.get("citations"):
                    with st.expander("View citations"):
                        for c in msg["citations"]:
                            st.markdown(f"- Source: {c.get('source','')}")
else:
    st.info("No index loaded. Upload a PDF and run processing, or run the preprocessing scripts manually.")
    st.markdown("**Manual steps (if you prefer):**")
    st.code("python process_document.py")
    st.code("python create_embeddings.py")
    st.code("streamlit run app.py")
