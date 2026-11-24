import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Auditable RAG Prototype", layout="wide")
st.title("🧠 Self-Correcting RAG Prototype")
st.caption("PDF retrieval + critic validation + query rewriting + cited answers")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Setup")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

uploaded_files = st.sidebar.file_uploader(
    "Upload clean PDF or text files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

enable_critic = st.sidebar.checkbox("Enable self-correction / critic", value=True)
k_chunks = st.sidebar.slider("Number of chunks to retrieve", min_value=2, max_value=8, value=4)
build_button = st.sidebar.button("Build Knowledge Index")

# -----------------------------
# Helper functions
# -----------------------------
def get_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def load_documents(files):
    all_docs = []
    for uploaded_file in files:
        filename = uploaded_file.name
        suffix = os.path.splitext(filename)[1].lower()
        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
            all_docs.extend(docs)
        elif suffix in [".txt", ".md"]:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            all_docs.append(Document(page_content=text, metadata={"source": filename}))
    return all_docs

def build_vector_index(files):
    docs = load_documents(files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks found. Your files may be empty or scanned images.")
    
    # FREE LOCAL EMBEDDINGS (No OpenAI key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore, len(docs), len(chunks)

def retrieve_chunks(vectorstore, query, k):
    return vectorstore.similarity_search(query, k=k)

def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        label = f"[{i}] Source: {source}, Page {page}" if page else f"[{i}] Source: {source}"
        blocks.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)

def critic_checks_context(question, context):
    llm = get_llm()
    prompt = f"""You are a strict retrieval quality checker.
Return only one word: YES if the context is sufficient to answer the question, NO if it is insufficient.
QUESTION: {question}
CONTEXT: {context}"""
    response = llm.invoke(prompt).content.strip()
    return response.upper().startswith("YES"), response

def rewrite_query(question):
    llm = get_llm()
    prompt = f"""Rewrite this question into a better, shorter search query for retrieving official policy documents.
Only output the rewritten search query.
Original question: {question}"""
    return llm.invoke(prompt).content.strip()

def generate_answer(question, context):
    llm = get_llm()
    prompt = f"""Answer the question using ONLY the provided sources.
Rules:
1. Every important factual claim must include an inline citation like [1] or [2].
2. Do not use outside knowledge.
3. If the sources do not contain the answer, say: "I do not have enough information in the provided documents."
QUESTION: {question}
SOURCES: {context}"""
    return llm.invoke(prompt).content.strip()

def run_rag_pipeline(question, vectorstore, enable_critic, k):
    steps = ["Router: Classified query as document-grounded question."]
    docs = retrieve_chunks(vectorstore, question, k)
    context = format_context(docs)
    steps.append(f"Retriever: Retrieved top {k} chunks.")
    
    if enable_critic:
        is_sufficient, verdict = critic_checks_context(question, context)
        steps.append(f"Critic: Context sufficient? {verdict}")
        if not is_sufficient:
            new_query = rewrite_query(question)
            steps.append(f"Critic: Rewrote query as: '{new_query}'")
            docs = retrieve_chunks(vectorstore, new_query, k)
            context = format_context(docs)
            steps.append("Retriever: Retrieved again using rewritten query.")
    else:
        steps.append("Critic: Disabled. Running baseline RAG.")
        
    answer = generate_answer(question, context)
    steps.append("Synthesizer: Generated final answer with citations.")
    return answer, docs, steps

# -----------------------------
# Build index
# -----------------------------
if build_button:
    if not api_key:
        st.sidebar.error("Please enter your Groq API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF, TXT, or MD file.")
    else:
        try:
            with st.spinner("Building vector index (this may take 10 seconds the first time)..."):
                vectorstore, num_docs, num_chunks = build_vector_index(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.num_docs = num_docs
                st.session_state.num_chunks = num_chunks
            st.sidebar.success(f"Indexed {num_docs} documents/pages into {num_chunks} chunks.")
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {str(e)}")

# -----------------------------
# Main UI
# -----------------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("Ask a question")
    question = st.text_input("Example: What is the withdrawal deadline?")
    if st.button("Ask"):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif "vectorstore" not in st.session_state:
            st.error("Please upload documents and click 'Build Knowledge Index' first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving, validating, and synthesizing..."):
                    answer, docs, steps = run_rag_pipeline(question, st.session_state.vectorstore, enable_critic, k_chunks)
                st.markdown("## Final Answer")
                st.markdown(answer)
                st.markdown("## Retrieved Sources")
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", None)
                    title = f"[{i}] {source}, Page {page}" if page else f"[{i}] {source}"
                    with st.expander(title):
                        st.write(doc.page_content)
                st.session_state.last_steps = steps
            except Exception as e:
                st.error(f"Application error: {str(e)}")

with right:
    st.subheader("Agent Telemetry")
    if "num_docs" in st.session_state:
        st.metric("Loaded documents/pages", st.session_state.num_docs)
        st.metric("Indexed chunks", st.session_state.num_chunks)
    if "last_steps" in st.session_state:
        for step in st.session_state.last_steps:
            st.write("✅ " + step)
    else:
        st.info("Agent steps will appear here after you ask a question.")