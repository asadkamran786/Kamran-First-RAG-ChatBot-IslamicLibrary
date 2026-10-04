
# ============================================================================
# NOOR LIBRARY - STREAMLIT CLOUD READY VERSION
# Fixed: file upload + Chroma persistence + requirements + dotenv + fitz
# ============================================================================
import os
import re
import hashlib
import tempfile
import uuid
from pathlib import Path
import json
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# --- FIX 1: dotenv optional on Cloud ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # On Streamlit Cloud we use st.secrets

# --- FIX 2: PyMuPDF with new API ---
try:
    import pymupdf
    HAS_PYMUPDF = True
    # Alias for old code
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

try:
    from gtts import gTTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

st.set_page_config(
    page_title="Noor Library - Islamic RAG Portal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GOLD DARK UI
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Amiri:wght@400;700&display=swap" rel="stylesheet">
<style>
.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(233,201,123,0.22), transparent 28%),
                radial-gradient(circle at 85% 12%, rgba(0,168,132,0.20), transparent 30%),
                linear-gradient(180deg, #081a24 0%, #06131a 100%);
    color: #ffffff !important;
}
html, body, [class*="css"] { color: #ffffff !important; font-family: 'Inter', sans-serif; }
p, span, div, label, li { color: #f5f5f5 !important; font-size: 15px !important; font-weight: 500 !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1e29 0%, #07161f 100%) !important; border-right: 2px solid rgba(233,201,123,0.25) !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
.hero { padding: 2rem 2.2rem; border-radius: 24px; background: linear-gradient(135deg, rgba(233,201,123,0.28) 0%, rgba(0,128,96,0.35) 100%); border: 2px solid rgba(233,201,123,0.35); margin-bottom: 1.5rem; }
.hero-kicker { color: #FFD700 !important; font-size: 0.85rem !important; font-weight: 800 !important; }
.hero h1 { font-size: clamp(2.2rem, 4.5vw, 3.8rem) !important; color: #ffffff !important; font-weight: 800 !important; }
.glass-card { background: rgba(18,50,58,0.95) !important; border: 2px solid rgba(233,201,123,0.25) !important; border-radius: 18px !important; padding: 1.3rem !important; }
.card-label { color: #FFD700 !important; font-weight: 800 !important; }
.card-value { font-size: 2rem !important; font-weight: 800 !important; color: #ffffff !important; }
.evidence-card { border-radius: 14px !important; padding: 1.2rem !important; background: rgba(255,255,255,0.08) !important; border-left: 5px solid #FFD700 !important; color: #ffffff !important; }
.source-chip { background: rgba(233,201,123,0.25) !important; border: 1.5px solid rgba(233,201,123,0.45) !important; color: #ffffff !important; font-weight: 700 !important; padding: .35rem .75rem !important; border-radius: 999px !important; }
.arabic-text { font-family: 'Amiri', serif !important; direction: rtl; font-size: 1.45rem !important; color: #FFECB3 !important; }
</style>
""", unsafe_allow_html=True)

# --- FIX 3: CLOUD PERSISTENCE - Use /tmp on Streamlit Cloud, local otherwise ---
# Streamlit Cloud is ephemeral, /tmp is writable. Also handle relative path.
if os.path.exists("/mount/src"):
    # We are on Streamlit Cloud
    PERSIST_DIR = Path(tempfile.gettempdir()) / "noor_library_chroma"
    IS_CLOUD = True
else:
    PERSIST_DIR = Path("./noor_library_chroma")
    IS_CLOUD = False

PERSIST_DIR.mkdir(parents=True, exist_ok=True)
INDEX_META_FILE = PERSIST_DIR / "indexed_books.json"
COLLECTION_NAME = "noor_islamic_library"

ROMAN_URDU_MAP = {
    'wazu': 'وضو', 'wudu': 'وضو', 'namaz': 'نماز', 'salah': 'صلاة', 'roza': 'روزہ',
    'hajj': 'حج', 'zakat': 'زکوۃ', 'dua': 'دعا', 'quran': 'قرآن', 'hadith': 'حدیث',
    'ghusl': 'غسل', 'azan': 'اذان', 'masjid': 'مسجد', 'halal': 'حلال', 'haram': 'حرام',
    'wuzu ke faraiz': 'وضو کے فرائض', 'namaz ka tarika': 'نماز کا طریقہ',
}

QURAN_REF_PATTERN = re.compile(r'(?:Surah|سورة|سورہ)?\s*([\w]+)?\s*(\d+):(\d+)', re.IGNORECASE)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

@st.cache_resource(show_spinner=False)
def get_llm():
    # FIX 4: Get API key from st.secrets on Cloud, else env
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return None
    
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.1, groq_api_key=api_key)

def load_index_meta():
    if INDEX_META_FILE.exists():
        try:
            return json.loads(INDEX_META_FILE.read_text())
        except:
            return {}
    return {}

def save_index_meta(meta):
    INDEX_META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def roman_urdu_to_urdu(text):
    lower = text.lower()
    expanded = lower
    for roman, urdu in ROMAN_URDU_MAP.items():
        if roman in lower:
            expanded = expanded.replace(roman, f"{roman} ({urdu})")
    return expanded

def detect_quran_refs(text):
    refs = []
    for match in QURAN_REF_PATTERN.finditer(text):
        surah = match.group(1) or ""
        ayah_num = match.group(2)
        verse_num = match.group(3)
        try:
            s = int(ayah_num)
            v = int(verse_num)
            if 1 <= s <= 114 and 1 <= v <= 300:
                refs.append({"text": f"{surah} {s}:{v}".strip(), "surah": s, "ayah": v, "url": f"https://quran.com/{s}/{v}"})
        except:
            continue
    return refs

def extract_text_pymupdf(pdf_path, book_title):
    docs = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and len(text.strip()) > 50:
                docs.append(Document(page_content=text, metadata={"source_file": book_title, "book_title": book_title, "page": page_num, "extraction": "pymupdf"}))
        doc.close()
    except Exception as e:
        st.error(f"PyMuPDF error: {e}")
    return docs

def get_vectorstore():
    embedding = load_embedding_model()
    # FIX: Use in-memory if persist fails on Cloud
    try:
        vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embedding, collection_name=COLLECTION_NAME)
        return vectorstore
    except Exception as e:
        st.warning(f"Using in-memory DB (Cloud ephemeral): {e}")
        return Chroma(embedding_function=embedding, collection_name=COLLECTION_NAME)

def main():
    st.markdown('<div class="hero"><div class="hero-kicker">نور لائبریری • NOOR LIBRARY</div><h1>Islamic RAG Portal — Gold Dark Premium</h1><p>Upload Islamic books (Quran, Hadith, Fiqh). Ask in English, Urdu, Arabic or Roman Urdu.</p></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Key")
        # Show if key exists
        has_key = False
        try:
            if st.secrets["GROQ_API_KEY"]:
                has_key = True
        except:
            has_key = bool(os.getenv("GROQ_API_KEY"))
        
        if has_key:
            st.success("GROQ_API_KEY found ✅")
        else:
            st.error("Add GROQ_API_KEY in Streamlit Secrets")
            st.code('GROQ_API_KEY = "gsk_..."')
            st.markdown("[Get free key](https://console.groq.com/keys)")
        
        if IS_CLOUD:
            st.info("☁️ Running on Streamlit Cloud — DB is temporary")
        else:
            st.info(f"📁 DB: {PERSIST_DIR}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        chunk_size = st.slider("Chunk size", 500, 2000, 1000)
        chunk_overlap = st.slider("Overlap", 50, 400, 200)
        top_k = st.slider("Top K sources", 1, 10, 5)
        search_mode = st.selectbox("Search mode", ["similarity", "mmr"])

    # Load vectorstore
    vectorstore = get_vectorstore()
    try:
        count = vectorstore._collection.count()
    except:
        count = 0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="glass-card"><div class="card-label">Total Chunks</div><div class="card-value">{count}</div></div>', unsafe_allow_html=True)
    with c2:
        meta = load_index_meta()
        st.markdown(f'<div class="glass-card"><div class="card-label">Books</div><div class="card-value">{len(meta)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="glass-card"><div class="card-label">Mode</div><div class="card-value" style="font-size:1.2rem">{search_mode}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="glass-card"><div class="card-label">Cloud</div><div class="card-value" style="font-size:1.2rem">{"Yes" if IS_CLOUD else "No"}</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload & Ask", "📚 Library"])

    with tab1:
        st.markdown("### 📤 Upload Islamic PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            # --- FIX 5: ROBUST FILE UPLOAD HANDLING ---
            all_docs = []
            meta = load_index_meta()
            
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                file_hash = get_file_hash(file_bytes)
                
                if file_hash in meta:
                    st.info(f"⏭️ Already indexed: {uploaded_file.name}")
                    continue
                
                # Use tempfile in system temp dir (writable on Cloud)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                try:
                    with st.spinner(f"Extracting {uploaded_file.name}..."):
                        if HAS_PYMUPDF:
                            docs = extract_text_pymupdf(tmp_path, uploaded_file.name)
                        else:
                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()
                        
                        if not docs:
                            st.warning(f"No text found in {uploaded_file.name} - scanned PDF needs OCR")
                        else:
                            # Chunk
                            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            chunked = splitter.split_documents(docs)
                            
                            # Add IDs
                            for doc in chunked:
                                doc.metadata["id"] = str(uuid.uuid4())
                            
                            all_docs.extend(chunked)
                            meta[file_hash] = {
                                "book_title": uploaded_file.name,
                                "chunks": len(chunked),
                                "pages": len(docs),
                                "hash": file_hash
                            }
                            st.success(f"✅ {uploaded_file.name}: {len(docs)} pages → {len(chunked)} chunks")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            if all_docs:
                with st.spinner(f"Indexing {len(all_docs)} chunks... This may take 1-2 min first time"):
                    try:
                        vectorstore.add_documents(all_docs)
                        save_index_meta(meta)
                        st.success(f"🎉 Indexed {len(all_docs)} chunks! Refresh page.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
                        st.code(str(e))

        st.markdown("---")
        st.markdown("### 💬 Ask Your Library")
        query = st.text_area("Your question", placeholder="e.g., wazu ke faraiz kya hain? | What are pillars of Wudu?", height=100)
        
        if st.button("🔍 Get Answer", type="primary") and query:
            llm = get_llm()
            if not llm:
                st.error("❌ GROQ_API_KEY missing. Add in Streamlit Cloud → Settings → Secrets")
                st.stop()
            
            expanded = roman_urdu_to_urdu(query)
            if expanded != query.lower():
                st.caption(f"🔎 Searching: {expanded}")
            
            with st.spinner("Searching..."):
                if search_mode == "mmr":
                    docs = vectorstore.max_marginal_relevance_search(query, k=top_k)
                else:
                    docs = vectorstore.similarity_search(query, k=top_k)
            
            if not docs:
                st.warning("No relevant documents found. Upload books first.")
            else:
                context = "\n\n".join([f"[Source: {d.metadata.get('book_title','Unknown')} Page {d.metadata.get('page',0)+1}]\n{d.page_content}" for d in docs])
                
                prompt = f"""You are Islamic scholar assistant. Answer ONLY from context.
Rules: Cite [Book, Page X]. Preserve Arabic exact. If not in context, say Not found in library.

QUESTION: {query}
EXPANDED: {expanded}
CONTEXT: {context}
Answer with citations:"""
                
                with st.spinner("Generating answer..."):
                    try:
                        answer = llm.invoke(prompt).content
                        st.markdown("#### 📖 Answer")
                        st.success(answer)
                        
                        # Quran refs
                        ans_refs = detect_quran_refs(answer)
                        if ans_refs:
                            st.markdown("#### 🔗 Quran References")
                            for ref in ans_refs:
                                st.markdown(f"- **{ref['text']}** → [quran.com]({ref['url']})")
                        
                        st.markdown("#### 📑 Evidence")
                        for i, d in enumerate(docs, 1):
                            src = d.metadata.get("book_title", "Unknown")
                            page = d.metadata.get("page", 0)+1
                            st.markdown(f'<div class="evidence-card"><span class="source-chip">{src}</span> <span class="source-chip">Page {page}</span><div style="margin-top:.6rem">{d.page_content[:800]}</div></div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"LLM Error: {e}")
                        if "decommissioned" in str(e) or "model" in str(e).lower():
                            st.info("Groq model changed. App uses openai/gpt-oss-20b now.")

    with tab2:
        st.markdown("### Library Manager")
        meta = load_index_meta()
        if not meta:
            st.info("No books indexed yet.")
        else:
            for h, info in meta.items():
                with st.expander(f"📘 {info['book_title']} — {info['chunks']} chunks"):
                    st.json(info)
        
        if st.button("🗑️ Clear Library"):
            try:
                vectorstore.delete_collection()
                if INDEX_META_FILE.exists():
                    INDEX_META_FILE.unlink()
                st.success("Cleared! Refresh page.")
            except Exception as e:
                st.error(f"Clear failed: {e}")

if __name__ == "__main__":
    main()
