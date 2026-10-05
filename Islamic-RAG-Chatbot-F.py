# ============================================================================
# KAMRAN ISLAMIC LIBRARY - FINAL ALL FIXES - Gold Dark Premium
# Fixes: detect_quran_refs, extract_text_pymupdf, Chroma Ephemeral, font 13px
# ============================================================================
import os, re, hashlib, tempfile, uuid, warnings, json
from pathlib import Path
from typing import List, Dict
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

try:
    import pymupdf
    HAS_PYMUPDF = True
    import pymupdf as fitz
except:
    try:
        import fitz
        HAS_PYMUPDF = True
    except:
        HAS_PYMUPDF = False

try:
    from gtts import gTTS
    HAS_TTS = True
except:
    HAS_TTS = False

try:
    from langchain_community.document_loaders import PyPDFLoader
except:
    PyPDFLoader = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

DISCLAIMER = "Disclaimer: For information only, not a Fatwa. Verify with authorized Ulama"
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME") is not None or "STREAMLIT" in os.environ or Path("/mount/src").exists()
PERSIST_DIR = Path(tempfile.gettempdir()) / "kamran_islamic_db" if IS_CLOUD else Path("./chroma_db")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "islamic_books"
INDEX_META_FILE = PERSIST_DIR / "index_meta.json"

QURAN_REF_PATTERN = re.compile(r'(?:Quran|Surah)\s*\d+:\d+|\[\d+:\d+\]', re.IGNORECASE)

def detect_quran_refs(text):
    try:
        return QURAN_REF_PATTERN.findall(text)
    except:
        return []

def extract_quran_refs(text):
    return detect_quran_refs(text)

def roman_urdu_to_urdu(text):
    mapping = {"wazu":"وضو","namaz":"نماز","roza":"روزہ","hajj":"حج","zakat":"زکوٰۃ","faraiz":"فرائض"}
    t = text.lower()
    for k,v in mapping.items():
        if k in t:
            t = t.replace(k, v)
    return t

def get_file_hash(b):
    return hashlib.md5(b).hexdigest()

def load_index_meta():
    if INDEX_META_FILE.exists():
        try:
            return json.loads(INDEX_META_FILE.read_text())
        except:
            return {}
    return {}

def save_index_meta(m):
    try:
        INDEX_META_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    except:
        pass

def extract_text_pymupdf(pdf_path, filename):
    docs = []
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            if txt and len(txt.strip()) > 20:
                docs.append(Document(page_content=txt.strip(), metadata={"source": filename, "page": i+1, "book_title": filename}))
        doc.close()
    except Exception as e:
        if PyPDFLoader:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for d in docs:
                d.metadata["book_title"] = filename
        else:
            raise e
    return docs

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    key = None
    try:
        key = st.secrets["GROQ_API_KEY"]
    except:
        key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=key, temperature=0.2)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    emb = load_embedding_model()
    try:
        if IS_CLOUD:
            import chromadb
            client = chromadb.EphemeralClient()
            return Chroma(client=client, embedding_function=emb, collection_name=COLLECTION_NAME)
        else:
            return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=emb, collection_name=COLLECTION_NAME)
    except:
        import chromadb, shutil
        try:
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        except:
            pass
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.EphemeralClient()
        return Chroma(client=client, embedding_function=emb, collection_name=COLLECTION_NAME)

st.set_page_config(page_title="Kamran Islamic Library", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(rgba(8,26,36,0.92), rgba(6,19,26,0.96)), url("https://images.unsplash.com/photo-1590075865003-1d9d56a0dcf3?q=80&w=2070") !important; background-size: cover !important; background-attachment: fixed !important; }
.hero { background: linear-gradient(135deg, rgba(18,50,58,0.95) 0%, rgba(8,26,36,0.98) 100%) !important; border: 1.5px solid rgba(255,215,0,0.3) !important; border-radius: 24px !important; padding: 28px !important; margin-bottom: 12px !important; }
.hero-kicker { color: #FFD700 !important; font-size: 36px !important; font-weight: 900 !important; }
.hero h1 { color: #fff !important; font-size: 2rem !important; font-weight: 800 !important; }
.hero p { color: #c8d6d6 !important; font-size: 14px !important; }
p, span, div, label, li { color: #f5f5f5 !important; font-size: 14px !important; }
.glass-card { background: rgba(18,50,58,0.95) !important; border: 1px solid rgba(255,215,0,0.25) !important; border-radius: 16px !important; padding: 12px !important; text-align: center !important; }
.evidence-card { background: rgba(18,50,58,0.9) !important; border-left: 4px solid #FFD700 !important; border-radius: 10px !important; padding: 14px !important; margin: 10px 0 !important; }
.source-chip { background: rgba(255,215,0,0.15) !important; border: 1px solid rgba(255,215,0,0.3) !important; color: #FFD700 !important; border-radius: 16px !important; padding: 3px 10px !important; font-size: 10px !important; margin: 3px !important; display: inline-block !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] label, [data-testid="stSidebar"] [data-testid="stSelectbox"] label { font-size: 13px !important; font-weight: 600 !important; }
[data-testid="stSidebar"] [data-testid="stThumbValue"] { font-size: 11px !important; }
.glass-card .card-label { font-size: 10px !important; color: #FFD700 !important; text-transform: uppercase !important; }
.glass-card .card-value { font-size: 1.1rem !important; font-weight: 800 !important; color: #fff !important; }
.stTextArea textarea { background: #fff !important; color: #000 !important; font-size: 15px !important; font-weight: 600 !important; border: 2px solid #FFD700 !important; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="hero"><div class="hero-kicker">Kamran Islamic Library</div><h1>Kamran Islamic RAG Portal — Gold Dark Premium</h1><p>Upload Islamic books (Quran, Hadith, Fiqh). Ask in English, Urdu, Arabic or Roman Urdu.</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background: rgba(255,215,0,0.12); border: 1.5px solid rgba(255,215,0,0.4); border-radius: 10px; padding: 6px; text-align:center;"><span style="color:#FFD700 !important; font-size:11px !important; font-weight:700 !important;">⚠️ {DISCLAIMER} | یہ جوابات صرف معلوماتی ہیں، فتویٰ نہیں</span></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🔑 API Key")
        has_key = False
        try:
            if st.secrets["GROQ_API_KEY"]:
                has_key = True
        except:
            has_key = bool(os.getenv("GROQ_API_KEY"))
        if has_key:
            st.success("GROQ_API_KEY ✅")
        else:
            st.error("Add GROQ_API_KEY in Secrets")
        st.info(f"Cloud: {'Yes' if IS_CLOUD else 'No'}")
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        chunk_size = st.slider("Chunk size", 500, 2000, 1000)
        chunk_overlap = st.slider("Overlap", 50, 400, 200)
        top_k = st.slider("Top K", 1, 10, 5)
        search_mode = st.selectbox("Search", ["similarity", "mmr"])

    vectorstore = get_vectorstore()
    try:
        count = vectorstore._collection.count()
    except:
        count = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="glass-card"><div class="card-label">Total Chunks</div><div class="card-value">{count}</div></div>', unsafe_allow_html=True)
    with c2:
        meta = load_index_meta()
        st.markdown(f'<div class="glass-card"><div class="card-label">Books</div><div class="card-value">{len(meta)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="glass-card"><div class="card-label">Mode</div><div class="card-value" style="font-size:1rem">{search_mode}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="glass-card"><div class="card-label">Cloud</div><div class="card-value" style="font-size:1rem">{"Yes" if IS_CLOUD else "No"}</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Upload & Ask", "Library"])
    with tab1:
        uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            all_docs = []
            meta = load_index_meta()
            for uf in uploaded_files:
                fb = uf.read()
                fh = get_file_hash(fb)
                if fh in meta:
                    st.info(f"Already: {uf.name}")
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as tmp:
                    tmp.write(fb)
                    tp = tmp.name
                try:
                    docs = extract_text_pymupdf(tp, uf.name)
                    if docs:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunked = splitter.split_documents(docs)
                        for d in chunked:
                            d.metadata["id"] = str(uuid.uuid4())
                        all_docs.extend(chunked)
                        meta[fh] = {"book_title": uf.name, "chunks": len(chunked), "pages": len(docs)}
                        st.success(f"✅ {uf.name}: {len(docs)} pages → {len(chunked)} chunks")
                finally:
                    try:
                        os.unlink(tp)
                    except:
                        pass
            if all_docs:
                with st.spinner(f"Indexing {len(all_docs)}..."):
                    vectorstore.add_documents(all_docs)
                    save_index_meta(meta)
                    st.success(f"🎉 Indexed {len(all_docs)}")
                    st.balloons()

        st.markdown("### 💬 Ask Your Library")
        query = st.text_area("Your question", placeholder="wazu ke faraiz? | What are pillars of Wudu?", height=90)
        if st.button("🔍 Get Answer", type="primary") and query:
            llm = get_llm()
            if not llm:
                st.error("❌ GROQ_API_KEY missing")
                st.stop()
            expanded = roman_urdu_to_urdu(query)
            with st.spinner(f"Searching {count} chunks..."):
                if search_mode == "mmr":
                    docs = vectorstore.max_marginal_relevance_search(expanded, k=top_k)
                else:
                    docs = vectorstore.similarity_search(expanded, k=top_k)
            if not docs:
                st.warning("No relevant content")
                st.stop()
            st.success(f"Found {len(docs)} references")
            ctx = ""
            for i, doc in enumerate(docs):
                src = doc.metadata.get('book_title', 'Unknown')
                page = doc.metadata.get('page', '?')
                ctx += f"\n[Source {i+1}: {src} Page {page}]\n{doc.page_content}\n"
            prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nExpanded: {expanded}\n\nAnswer in same language. Quote refs. If not found say not found. {DISCLAIMER}\n\nAnswer:"
            with st.spinner("Generating..."):
                resp = llm.invoke(prompt)
                st.markdown("### 📖 Answer")
                st.markdown(f'<div class="evidence-card">{resp.content}</div>', unsafe_allow_html=True)
                for i, doc in enumerate(docs):
                    st.markdown(f'<span class="source-chip">{doc.metadata.get("book_title","")} Page {doc.metadata.get("page","?")}</span>', unsafe_allow_html=True)

    with tab2:
        meta = load_index_meta()
        for info in meta.values():
            st.markdown(f'<div class="glass-card" style="text-align:left"><b>{info["book_title"]}</b> - {info["chunks"]} chunks</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
