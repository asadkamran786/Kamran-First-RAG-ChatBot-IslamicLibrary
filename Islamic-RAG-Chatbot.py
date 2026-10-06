# ============================================================================
# KAMRAN ISLAMIC LIBRARY - FINAL WORKING 2025 - ISLAMIC + TECHNICAL GOLD DARK
# Fixes: font 14px, detect_quran_refs, Chroma Ephemeral, Groq 2025 models, bg
# ============================================================================
import os, re, hashlib, tempfile, uuid, warnings, json
from pathlib import Path
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
    import pymupdf as fitz
    HAS_PYMUPDF = True
except:
    try:
        import fitz
        HAS_PYMUPDF = True
    except:
        HAS_PYMUPDF = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
try:
    from langchain_community.document_loaders import PyPDFLoader
except:
    PyPDFLoader = None

DISCLAIMER = "Disclaimer: For information only, not a Fatwa. Verify with Ulama"
IS_CLOUD = os.getenv("STREAMLIT_RUNTIME") is not None or "STREAMLIT" in os.environ or Path("/mount/src").exists()
PERSIST_DIR = Path(tempfile.gettempdir()) / "kamran_islamic_db" if IS_CLOUD else Path("./chroma_db")
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "islamic_books"
INDEX_META_FILE = PERSIST_DIR / "index_meta.json"
QURAN_REF_PATTERN = re.compile(r'(?:Quran|Surah)\s*\d+:\d+|\[\d+:\d+\]', re.IGNORECASE)

def detect_quran_refs(text):
    try: return QURAN_REF_PATTERN.findall(text)
    except: return []
def extract_quran_refs(text): return detect_quran_refs(text)
def roman_urdu_to_urdu(text):
    m={"wazu":"وضو","namaz":"نماز","roza":"روزہ","hajj":"حج","zakat":"زکوٰۃ","faraiz":"فرائض"}
    t=text.lower()
    for k,v in m.items():
        if k in t: t=t.replace(k,v)
    return t
def get_file_hash(b): return hashlib.md5(b).hexdigest()
def load_index_meta():
    if INDEX_META_FILE.exists():
        try: return json.loads(INDEX_META_FILE.read_text())
        except: return {}
    return {}
def save_index_meta(m):
    try: INDEX_META_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    except: pass
def extract_text_pymupdf(pdf_path, filename):
    docs=[]
    try:
        doc=fitz.open(pdf_path)
        for i,page in enumerate(doc):
            txt=page.get_text("text")
            if txt and len(txt.strip())>20:
                docs.append(Document(page_content=txt.strip(), metadata={"source":filename,"page":i+1,"book_title":filename}))
        doc.close()
    except Exception as e:
        if PyPDFLoader:
            loader=PyPDFLoader(pdf_path)
            docs=loader.load()
            for d in docs: d.metadata["book_title"]=filename
        else:
            raise e
    return docs

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    key=None
    try: key=st.secrets["GROQ_API_KEY"]
    except: key=os.getenv("GROQ_API_KEY")
    if not key: return None
    # 2025 WORKING GROQ MODELS - in order of preference
    for model_name in ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct"]:
        try:
            return ChatGroq(model=model_name, groq_api_key=key, temperature=0.2)
        except:
            continue
    # final fallback
    return ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key, temperature=0.2)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    emb=load_embedding_model()
    try:
        if IS_CLOUD:
            import chromadb
            client=chromadb.EphemeralClient()
            return Chroma(client=client, embedding_function=emb, collection_name=COLLECTION_NAME)
        else:
            return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=emb, collection_name=COLLECTION_NAME)
    except:
        import chromadb, shutil
        try: shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        except: pass
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client=chromadb.EphemeralClient()
        return Chroma(client=client, embedding_function=emb, collection_name=COLLECTION_NAME)

st.set_page_config(page_title="Kamran RAG Chatbot-Islamic Library", page_icon="🕌", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Amiri:wght@700&display=swap" rel="stylesheet">
<style>
.stApp {
    background: 
        radial-gradient(circle at 20% 30%, rgba(255,215,0,0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(0,255,200,0.06) 0%, transparent 50%),
        linear-gradient(135deg, rgba(5,15,22,0.94) 0%, rgba(8,26,36,0.96) 30%, rgba(12,38,52,0.93) 70%, rgba(6,20,30,0.95) 100%),
        url("https://images.unsplash.com/photo-1585036156171-48532484fbb1?q=80&w=2070") !important;
    background-size: cover !important;
    background-attachment: fixed !important;
    background-blend-mode: overlay, overlay, normal, normal !important;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(255,215,0,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,215,0,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: -1;
}
.hero {
    background: 
        linear-gradient(135deg, rgba(255,215,0,0.08) 0%, rgba(0,0,0,0.2) 100%),
        linear-gradient(135deg, rgba(18,50,58,0.98) 0%, rgba(8,26,36,0.98) 50%, rgba(15,45,60,0.96) 100%),
        url("https://images.unsplash.com/photo-1590075865003-1d9d56a0dcf3?q=80&w=2070") !important;
    background-size: cover !important;
    background-blend-mode: overlay, normal, normal !important;
    border: 2px solid rgba(255,215,0,0.35) !important;
    border-radius: 28px !important;
    padding: 36px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 30px rgba(255,215,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "﷽";
    position: absolute;
    top: 10px;
    right: 30px;
    font-family: 'Amiri', serif;
    font-size: 42px;
    color: rgba(255,215,0,0.15);
    pointer-events: none;
}
.hero-kicker { color: #FFD700 !important; font-size: 36px !important; font-weight: 900 !important; letter-spacing: 2px !important; text-shadow: 0 2px 10px rgba(255,215,0,0.4) !important; }
.hero h1 { color: #ffffff !important; font-size: 2.3rem !important; font-weight: 900 !important; margin: 10px 0 !important; text-shadow: 0 2px 20px rgba(0,0,0,0.5) !important; }
.hero p { color: #a8c0c0 !important; font-size: 15px !important; margin-top: 10px !important; }
p, span, div, label, li { color: #f5f5f5 !important; font-size: 14px !important; }
.glass-card { background: linear-gradient(135deg, rgba(18,50,58,0.98) 0%, rgba(12,38,50,0.95) 100%) !important; border: 1px solid rgba(255,215,0,0.3) !important; border-radius: 16px !important; padding: 14px !important; text-align: center !important; box-shadow: 0 4px 15px rgba(0,0,0,0.3), 0 0 10px rgba(255,215,0,0.1) !important; }
.evidence-card { background: linear-gradient(135deg, rgba(18,50,58,0.95) 0%, rgba(10,35,45,0.9) 100%) !important; border-left: 4px solid #FFD700 !important; border-top: 1px solid rgba(255,215,0,0.2) !important; border-radius: 12px !important; padding: 16px !important; margin: 12px 0 !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important; }
.source-chip { background: rgba(255,215,0,0.15) !important; border: 1px solid rgba(255,215,0,0.35) !important; color: #FFD700 !important; border-radius: 20px !important; padding: 4px 12px !important; font-size: 10px !important; margin: 3px !important; display: inline-block !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] label, [data-testid="stSidebar"] [data-testid="stSelectbox"] label { font-size: 13px !important; font-weight: 600 !important; color: #FFD700 !important; }
[data-testid="stSidebar"] [data-testid="stThumbValue"] { font-size: 11px !important; }
.glass-card .card-label { font-size: 10px !important; color: #FFD700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
.glass-card .card-value { font-size: 1.1rem !important; font-weight: 800 !important; color: #fff !important; }
.stTextArea textarea { background: #ffffff !important; color: #000 !important; font-size: 15px !important; font-weight: 600 !important; border: 2px solid #FFD700 !important; border-radius: 12px !important; }
.stButton button { background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important; color: #000 !important; font-weight: 800 !important; border-radius: 12px !important; border: none !important; box-shadow: 0 4px 15px rgba(255,215,0,0.3) !important; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="hero"><div class="hero-kicker">Kamran Islamic Library</div><h1>🕌 Kamran Islamic RAG Portal — Gold Dark Premium</h1><p>✨ Islamic Knowledge + AI Technology • Quran • Hadith • Fiqh • 5106+ Chunks</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background: linear-gradient(90deg, rgba(255,215,0,0.15), rgba(255,165,0,0.1)); border: 1.5px solid rgba(255,215,0,0.4); border-radius: 12px; padding: 8px; text-align:center;"><span style="color:#FFD700 !important; font-size:11px !important; font-weight:700 !important;">⚠️ {DISCLAIMER} | یہ جوابات صرف معلوماتی ہیں، فتویٰ نہیں</span></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🔑 API")
        has_key=False
        try:
            if st.secrets["GROQ_API_KEY"]: has_key=True
        except:
            has_key=bool(os.getenv("GROQ_API_KEY"))
        if has_key: st.success("GROQ ✅")
        else: st.error("Add GROQ_API_KEY")
        st.info(f"Cloud: {'Yes' if IS_CLOUD else 'No'}")
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        chunk_size=st.slider("Chunk size", 500, 2000, 1000)
        chunk_overlap=st.slider("Overlap", 50, 400, 200)
        top_k=st.slider("Top K", 1, 10, 5)
        search_mode=st.selectbox("Search", ["similarity","mmr"])

    vs=get_vectorstore()
    try: cnt=vs._collection.count()
    except: cnt=0
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f'<div class="glass-card"><div class="card-label">Total Chunks</div><div class="card-value">{cnt}</div></div>', unsafe_allow_html=True)
    with c2:
        meta=load_index_meta()
        st.markdown(f'<div class="glass-card"><div class="card-label">Books</div><div class="card-value">{len(meta)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="glass-card"><div class="card-label">Mode</div><div class="card-value" style="font-size:1rem">{search_mode}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="glass-card"><div class="card-label">Cloud</div><div class="card-value" style="font-size:1rem">{"Yes" if IS_CLOUD else "No"}</div></div>', unsafe_allow_html=True)

    t1,t2=st.tabs(["📤 Upload & Ask","📚 Library"])
    with t1:
        ups=st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if ups:
            all_docs=[]
            meta=load_index_meta()
            for uf in ups:
                fb=uf.read()
                fh=get_file_hash(fb)
                if fh in meta:
                    st.info(f"Already: {uf.name}")
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as tmp:
                    tmp.write(fb)
                    tp=tmp.name
                try:
                    docs=extract_text_pymupdf(tp, uf.name)
                    if docs:
                        sp=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        ch=sp.split_documents(docs)
                        for d in ch: d.metadata["id"]=str(uuid.uuid4())
                        all_docs.extend(ch)
                        meta[fh]={"book_title":uf.name,"chunks":len(ch),"pages":len(docs)}
                        st.success(f"✅ {uf.name}: {len(docs)} pages → {len(ch)} chunks")
                finally:
                    try: os.unlink(tp)
                    except: pass
            if all_docs:
                with st.spinner(f"Indexing {len(all_docs)}..."):
                    vs.add_documents(all_docs)
                    save_index_meta(meta)
                    st.success(f"🎉 Indexed {len(all_docs)}")
                    st.balloons()
        st.markdown("### 💬 Ask Your Library")
        q=st.text_area("Question", placeholder="wazu ke faraiz? | What are pillars of Wudu?", height=90)
        if st.button("🔍 Get Answer", type="primary") and q:
            llm=get_llm()
            if not llm:
                st.error("GROQ_API_KEY missing")
                st.stop()
            exp=roman_urdu_to_urdu(q)
            with st.spinner(f"Searching {cnt} chunks..."):
                docs=vs.max_marginal_relevance_search(exp,k=top_k) if search_mode=="mmr" else vs.similarity_search(exp,k=top_k)
            if not docs:
                st.warning("No content")
                st.stop()
            st.success(f"Found {len(docs)}")
            ctx=""
            for i,doc in enumerate(docs):
                src=doc.metadata.get('book_title','Unknown')
                pg=doc.metadata.get('page','?')
                ctx+=f"\n[Source {i+1}: {src} Page {pg}]\n{doc.page_content}\n"
            prompt=f"Context:\n{ctx}\n\nQuestion: {q}\nExpanded: {exp}\n\nAnswer in same language. Quote refs. Mention source & page. If not found say not found. {DISCLAIMER}\n\nAnswer:"
            with st.spinner("Generating..."):
                try:
                    resp=llm.invoke(prompt)
                    st.markdown("### 📖 Answer")
                    st.markdown(f'<div class="evidence-card">{resp.content}</div>', unsafe_allow_html=True)
                    for i,doc in enumerate(docs):
                        st.markdown(f'<span class="source-chip">{doc.metadata.get("book_title","")} Page {doc.metadata.get("page","?")}</span>', unsafe_allow_html=True)
                        with st.expander(f"Source {i+1}"): st.text(doc.page_content[:1000])
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    st.code(str(e))
    with t2:
        meta=load_index_meta()
        for info in meta.values():
            st.markdown(f'<div class="glass-card" style="text-align:left"><b>{info["book_title"]}</b> - {info["chunks"]} chunks</div>', unsafe_allow_html=True)

if __name__=="__main__":
    main()
