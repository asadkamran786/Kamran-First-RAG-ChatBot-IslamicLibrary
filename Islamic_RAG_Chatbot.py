# Noor Library - Complete Islamic RAG Portal with Roman Urdu, Quran Detector & Audio
# Built on top of Aicademy 360 RAG logic + new Islamic optimizations
import os  # For environment variables and file paths
import re  # For regex pattern matching (Quran refs, Roman Urdu)
import hashlib  # For MD5 hash deduplication of books
import tempfile  # For safe temporary PDF handling
import uuid  # For unique chunk IDs to avoid collisions
from pathlib import Path  # For clean path management
import json  # For persistent index metadata storage
from typing import List, Dict  # For type hints
import streamlit as st  # For web app UI
from dotenv import load_dotenv  # For loading GROQ_API_KEY from .env
from langchain_community.document_loaders import PyPDFLoader  # Fallback PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For smart chunking with overlap
from langchain_huggingface import HuggingFaceEmbeddings  # For local multilingual embeddings
from langchain_chroma import Chroma  # For persistent vector database
from langchain_groq import ChatGroq  # For fast LLM inference
from langchain_core.documents import Document  # For document wrapper with metadata

# Try importing PyMuPDF for fast text extraction (10x faster than PyPDF)
try:
    import fitz  # PyMuPDF library
    HAS_PYMUPDF = True  # Flag to enable fast path
except ImportError:
    HAS_PYMUPDF = False  # Fallback to PyPDFLoader

# Try importing OCR libraries for scanned image PDFs (Arabic/Urdu/English)
try:
    import pytesseract  # OCR engine
    from PIL import Image  # Image processing for OCR
    HAS_OCR = True  # Flag to enable OCR path
except ImportError:
    HAS_OCR = False  # Text-only mode if OCR not installed

# Try importing audio library for Arabic recitation playback
try:
    from gtts import gTTS  # Google Text-to-Speech
    HAS_TTS = True  # Flag to enable audio
except ImportError:
    HAS_TTS = False  # Disable audio if not installed

load_dotenv()  # Load .env file for API keys

# Configure Streamlit page with Islamic theme
st.set_page_config(
    page_title="Kamran Islamic Library - RAG Portal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for Islamic premium look (same glass-card logic as original)
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 0%, rgba(0, 168, 132, 0.18), transparent 28%),
                radial-gradient(circle at 85% 12%, rgba(212, 175, 55, 0.15), transparent 25%),
                #06131a;
    color: #eef6f1;
}
[data-testid="stSidebar"] { background: rgba(7, 28, 32, 0.98); border-right: 1px solid rgba(212,175,55,0.15); }
.hero {
    padding: 1.6rem 1.8rem; border-radius: 26px;
    background: linear-gradient(135deg, rgba(0,128,96,.35), rgba(212,175,55,.18));
    border: 1px solid rgba(212,175,55,0.18); margin-bottom: 1.2rem;
}
.hero-kicker { color: #d4af37; font-size: 0.8rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
.hero h1 { margin: .2rem 0 .5rem 0; font-size: clamp(2.1rem, 4vw, 3.8rem); color: white; }
.hero p { color: #b8d4c8; font-size: 1.05rem; max-width: 900px; }
.glass-card { background: rgba(12, 38, 42, 0.72); border: 1px solid rgba(212,175,55,0.10); border-radius: 20px; padding: 1.1rem; }
.card-label { color: #8fbfa8; font-size: .76rem; text-transform: uppercase; font-weight: 700; }
.card-value { font-size: 1.7rem; font-weight: 800; color: white; margin-top: .25rem; }
.card-sub { color: #7aa898; font-size: .82rem; }
.evidence-card { border-radius: 16px; padding: 1rem; margin: .6rem 0; background: rgba(255,255,255,.04); border-left: 4px solid #d4af37; }
.source-chip { display: inline-block; padding: .22rem .6rem; border-radius: 999px; margin-right: .35rem; background: rgba(212,175,55,.14); border: 1px solid rgba(212,175,55,.28); color: #f0d78c; font-size: .74rem; font-weight: 700; }
.arabic-text { font-family: 'Amiri', serif; direction: rtl; font-size: 1.25rem; line-height: 2; }
.quran-ref { background: rgba(0,168,132,0.15); border: 1px solid rgba(0,168,132,0.3); padding: 2px 8px; border-radius: 8px; color: #7CFFB2; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Define persistent storage paths (same optimization as original but persistent)
PERSIST_DIR = Path("./noor_library_chroma")  # Directory to store ChromaDB on disk
PERSIST_DIR.mkdir(exist_ok=True)  # Create directory if not exists
INDEX_META_FILE = PERSIST_DIR / "indexed_books.json"  # File to track indexed books with hash
COLLECTION_NAME = "noor_islamic_library"  # Chroma collection name

# Roman Urdu to Urdu/Arabic mapping for cross-language search
ROMAN_URDU_MAP = {
    'wazu': 'وضو', 'wudu': 'وضو', 'namaz': 'نماز', 'salah': 'صلاة', 'salat': 'صلاة',
    'roza': 'روزہ', 'saum': 'صوم', 'hajj': 'حج', 'zakat': 'زکوۃ', 'niyya': 'نیت', 'niyyah': 'نیت',
    'dua': 'دعا', 'sunnat': 'سنت', 'farz': 'فرض', 'wajib': 'واجب', 'halal': 'حلال', 'haram': 'حرام',
    'quran': 'قرآن', 'hadith': 'حدیث', 'hadees': 'حدیث', 'tafsir': 'تفسیر', 'fiqh': 'فقہ',
    'ghusl': 'غسل', 'tayammum': 'تیمم', 'azan': 'اذان', 'masjid': 'مسجد', 'iman': 'ایمان',
    'islam': 'اسلام', 'deeen': 'دین', 'ilm': 'علم', 'sabr': 'صبر', 'shukr': 'شکر',
    'wuzu ke faraiz': 'وضو کے فرائض', 'namaz ka tarika': 'نماز کا طریقہ', 'roze ke ahkam': 'روزے کے احکام'
}

# Quran reference regex to auto-detect ayah citations
QURAN_REF_PATTERN = re.compile(r'(?:Surah|سورة|سورہ)?\s*([\w]+)?\s*(\d+):(\d+)', re.IGNORECASE)

@st.cache_resource(show_spinner=False)  # Cache embedding model (original optimization: load once)
def load_embedding_model():
    """Load the local embedding model once and reuse it across Streamlit reruns."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual model for Arabic/Urdu/English/Roman Urdu
        model_kwargs={"device": "cpu"},  # Use CPU for compatibility
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},  # Batch optimization for speed
    )

# === GROQ ACTIVE FREE MODELS - Aug 2026 ===
# Llama models are now Enterprise-only. Free replacement: openai/gpt-oss-20b
GROQ_MODELS = [
    "openai/gpt-oss-20b",      # FASTEST FREE - 20B params - replacement for llama-3.1-8b-instant
    "openai/gpt-oss-120b",     # BEST QUALITY FREE - 120B params - replacement for llama-3.3-70b-versatile
]

@st.cache_resource(show_spinner=False)
def get_llm():
    """Load Groq LLM - uses FREE active model"""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        max_tokens=2000,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def invoke_with_fallback(prompt: str):
    """Try FREE active models - auto handles 404/Enterprise errors"""
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0.1,
                max_tokens=2000,
                api_key=os.getenv("GROQ_API_KEY"),
            )
            resp = llm.invoke(prompt)
            st.toast(f"✅ Model: {model_name}", icon="🤖")
            return resp
        except Exception as e:
            last_error = e
            continue
    raise last_error


def load_index_meta():  # Load indexed books metadata from disk
    if INDEX_META_FILE.exists():  # Check if meta file exists
        try:
            return json.loads(INDEX_META_FILE.read_text())  # Parse JSON
        except:
            return {}  # Return empty if corrupt
    return {}  # Return empty for first run

def save_index_meta(meta):  # Save metadata to disk
    INDEX_META_FILE.write_text(json.dumps(meta, indent=2))  # Write pretty JSON

def file_hash(file_bytes: bytes) -> str:  # Generate MD5 hash for deduplication
    return hashlib.md5(file_bytes).hexdigest()  # Return hash string

def normalize_arabic(text: str) -> str:  # Normalize Arabic for better retrieval
    if not text:  # Handle empty text
        return text  # Return as is
    text = re.sub(r'\u0640', '', text)  # Remove tatweel (ـ)
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef variants
    text = re.sub(r'ة', 'ه', text)  # Normalize Teh Marbuta
    return text  # Return normalized text

def roman_urdu_to_urdu(query: str) -> str:  # Convert Roman Urdu query to Urdu script
    query_lower = query.lower()  # Lowercase for matching
    expanded = query_lower  # Start with original query
    for roman, urdu in ROMAN_URDU_MAP.items():  # Loop through mapping dictionary
        if roman in query_lower:  # If roman word found in query
            expanded += f" {urdu}"  # Append Urdu equivalent for hybrid search
    return expanded  # Return expanded query

def detect_quran_refs(text: str) -> List[Dict]:  # Detect Quran references like 2:255
    refs = []  # List to store detected refs
    for match in QURAN_REF_PATTERN.finditer(text):  # Find all matches
        surah, ayah = match.group(2), match.group(3)  # Extract surah and ayah numbers
        try:
            s_num, a_num = int(surah), int(ayah)  # Convert to integers
            if 1 <= s_num <= 114 and 1 <= a_num <= 286:  # Validate Quran range
                refs.append({  # Add valid ref
                    "surah": s_num,
                    "ayah": a_num,
                    "url": f"https://quran.com/{s_num}/{a_num}",  # Generate quran.com link
                    "text": match.group(0)  # Original matched text
                })
        except:
            continue  # Skip invalid numbers
    return refs  # Return list of refs

def generate_audio(text: str, lang: str = 'ar') -> str:  # Generate audio from Arabic/Urdu text
    if not HAS_TTS:  # Check if TTS library available
        return None  # Return None if not available
    try:
        clean_text = text[:500]  # Limit to 500 chars for TTS
        tts = gTTS(text=clean_text, lang=lang, slow=False)  # Create TTS object
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name  # Temp file
        tts.save(temp_path)  # Save audio
        return temp_path  # Return path to audio file
    except Exception as e:
        return None  # Return None on error

def extract_text_hybrid(pdf_path: str, filename: str) -> List[Document]:  # Hybrid extractor: text + OCR
    docs = []  # List to store documents
    if HAS_PYMUPDF:  # Use fast PyMuPDF if available
        doc = fitz.open(pdf_path)  # Open PDF with PyMuPDF
        for i, page in enumerate(doc):  # Iterate pages
            text = page.get_text("text").strip()  # Extract text fast
            if len(text) > 80:  # If enough text, it's a text PDF
                docs.append(Document(
                    page_content=text,  # Use extracted text
                    metadata={"source_file": filename, "page": i, "extraction": "text", "book": filename}
                ))
            else:  # Scanned image page - needs OCR
                if HAS_OCR:  # Check if OCR available
                    try:
                        pix = page.get_pixmap(dpi=200)  # Render page to image at 200 DPI (speed/quality balance)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert to PIL
                        ocr_text = pytesseract.image_to_string(img, lang='ara+urd+eng')  # OCR with 3 languages
                        if len(ocr_text.strip()) > 20:  # If OCR got meaningful text
                            docs.append(Document(
                                page_content=ocr_text,
                                metadata={"source_file": filename, "page": i, "extraction": "ocr", "book": filename}
                            ))
                    except:
                        if text:  # Fallback to original text
                            docs.append(Document(page_content=text, metadata={"source_file": filename, "page": i, "extraction": "fallback", "book": filename}))
        doc.close()  # Close PDF
    else:  # Fallback to original PyPDFLoader logic
        pages = PyPDFLoader(pdf_path).load()  # Load with LangChain
        for p in pages:  # Add metadata
            p.metadata["source_file"] = filename
            p.metadata["book"] = filename
            p.metadata["extraction"] = "pypdf"
        docs = pages  # Assign
    return docs  # Return all page documents

def build_knowledge_base(files, chunk_size, overlap, progress_bar=None):  # Build/update KB (core optimization from original)
    """Read PDFs, add source metadata, split them, embed them, and create Chroma - with hash dedup and persistence."""
    indexed_meta = load_index_meta()  # Load existing index metadata
    embedding = load_embedding_model()  # Load cached embedding model
    
    # Try loading existing vectorstore from disk (persistent optimization)
    vectorstore = None  # Initialize
    if any(PERSIST_DIR.iterdir()):  # If persist dir not empty
        try:
            vectorstore = Chroma(
                persist_directory=str(PERSIST_DIR),  # Load from disk
                embedding_function=embedding,  # Use same embedding
                collection_name=COLLECTION_NAME,  # Same collection
            )
        except:
            vectorstore = None  # Reset if load fails

    all_new_chunks = []  # Collect new chunks to embed
    total_pages = 0  # Track pages
    newly_indexed = []  # Track new books
    skipped = []  # Track skipped duplicates

    for idx, uploaded_pdf in enumerate(files):  # Iterate uploaded files
        if progress_bar:  # Update progress if provided
            progress_bar.progress(idx/len(files), text=f"Checking {uploaded_pdf.name}...")
        
        fhash = file_hash(uploaded_pdf.getvalue())  # Calculate file hash
        if fhash in indexed_meta:  # If already indexed
            skipped.append(uploaded_pdf.name)  # Skip to save time
            continue  # Next file

        # Write to temp file (same safe pattern as original)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")  # Create temp file
        temp_file.write(uploaded_pdf.getvalue())  # Write PDF bytes
        temp_file.close()  # Close file

        try:
            pages = extract_text_hybrid(temp_file.name, uploaded_pdf.name)  # Extract with hybrid OCR
            total_pages += len(pages)  # Count pages

            book_title = Path(uploaded_pdf.name).stem  # Extract book title from filename
            for page in pages:  # Add metadata to each page
                page.metadata["book_title"] = book_title  # Add title
                page.metadata["file_hash"] = fhash  # Add hash for dedup
                page.page_content = normalize_arabic(page.page_content)  # Normalize Arabic

            # Smart chunking with Arabic punctuation (optimization over original)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,  # From slider
                chunk_overlap=overlap,  # From slider
                separators=["\n\n", "\n", ".", "۔", "؟", "!", " ", ""],  # Include Urdu/Arabic separators
            )
            chunks = splitter.split_documents(pages)  # Split into chunks

            for c in chunks:  # Add unique IDs to chunks
                c.metadata["chunk_id"] = f"{fhash}_{c.metadata.get('page',0)}_{uuid.uuid4().hex[:6]}"  # Unique ID
            
            all_new_chunks.extend(chunks)  # Add to batch

            # Save metadata for incremental tracking
            indexed_meta[fhash] = {
                "filename": uploaded_pdf.name,
                "book_title": book_title,
                "pages": len(pages),
                "chunks": len(chunks),
                "indexed_at": str(Path(temp_file.name).stat().st_mtime)
            }
            newly_indexed.append(uploaded_pdf.name)  # Track new book
            
        finally:
            os.unlink(temp_file.name)  # Always delete temp file (original pattern)

    # If no new chunks but existing store exists, return existing
    if not all_new_chunks and vectorstore:
        return vectorstore, 0, 0, skipped, newly_indexed, indexed_meta  # Return existing

    # If no new chunks and no existing store, return None
    if not all_new_chunks and not vectorstore:
        return None, 0, 0, skipped, newly_indexed, indexed_meta

    # Batch embedding ingestion (performance optimization)
    if progress_bar:
        progress_bar.progress(0.7, text=f"Embedding {len(all_new_chunks)} chunks (batch=64)...")

    if vectorstore is None:  # First time - create new collection
        vectorstore = Chroma.from_documents(
            documents=all_new_chunks,  # New chunks
            embedding=embedding,  # Cached embedding model
            persist_directory=str(PERSIST_DIR),  # Persist to disk
            collection_name=COLLECTION_NAME,  # Collection name
        )
    else:  # Incremental add (don't re-index old books)
        vectorstore.add_documents(all_new_chunks)  # Add only new

    save_index_meta(indexed_meta)  # Save updated metadata
    if progress_bar:
        progress_bar.progress(1.0, text="Index ready!")  # Complete
    
    return vectorstore, total_pages, len(all_new_chunks), skipped, newly_indexed, indexed_meta

def retrieve_documents(vectorstore, query, search_mode, top_k):  # Retrieve with multiple strategies (original logic + Roman Urdu)
    """Retrieve document chunks using either similarity search or MMR - enhanced with Roman Urdu."""
    expanded_query = roman_urdu_to_urdu(query)  # Expand Roman Urdu to Urdu script
    normalized_query = normalize_arabic(expanded_query)  # Normalize Arabic

    if search_mode == "MMR - diverse evidence":  # Diverse retrieval (original logic)
        return vectorstore.max_marginal_relevance_search(
            normalized_query,
            k=top_k,
            fetch_k=max(20, top_k * 4),  # Fetch more for diversity
        )
    elif search_mode == "Similarity":  # Simple similarity (original logic)
        return vectorstore.similarity_search(normalized_query, k=top_k)
    else:  # Hybrid - best for Islamic books (new optimization)
        docs = vectorstore.similarity_search(normalized_query, k=top_k*2)  # Fetch double
        tokens = set(re.findall(r'\w+', normalized_query.lower()))  # Tokenize query
        def score(d):  # Keyword overlap scoring
            content_tokens = set(re.findall(r'\w+', d.page_content.lower()))
            return len(tokens & content_tokens)  # Overlap count
        docs = sorted(docs, key=score, reverse=True)[:top_k]  # Re-rank and cut
        return docs

def create_context(docs, max_chars=14000):  # Create labeled context for LLM (original pattern)
    """Turn retrieved chunks into one labeled context block for the LLM."""
    parts = []  # List of context parts
    used = 0  # Track chars used
    for i, doc in enumerate(docs, start=1):  # Enumerate docs
        source = doc.metadata.get("book_title", doc.metadata.get("source_file", "Unknown"))  # Get source
        page = doc.metadata.get("page", 0) + 1  # Get page (1-indexed)
        extraction = doc.metadata.get("extraction", "text")  # Get extraction type
        chunk = f"[Source {i}: {source} | Page {page} | {extraction}]\n{doc.page_content}\n"  # Format chunk
        if used + len(chunk) > max_chars:  # Check char limit
            break  # Stop if exceeds
        parts.append(chunk)  # Add chunk
        used += len(chunk)  # Update counter
    return "\n---\n".join(parts)  # Join with separator

# Sidebar - same structure as original portal
with st.sidebar:
    st.markdown("### 📚 Noor Library Setup")  # Sidebar title
    st.caption("Upload Islamic books - supports Roman Urdu search")  # Helper text
    
    uploaded_files = st.file_uploader(  # File uploader (original logic)
        "Upload books",
        type=["pdf"],
        accept_multiple_files=True,
        help="Sahih Bukhari, Tafsir, Fiqh, Seerah - Text or scanned"
    )
    
    st.divider()  # Divider
    st.markdown("#### ⚙️ RAG Settings")  # Settings header
    chunk_size = st.slider("Chunk Size", 400, 1200, 800, step=50, help="800 optimal for Arabic/Urdu")  # Chunk size slider
    overlap = st.slider("Overlap", 50, 300, 150, step=10)  # Overlap slider
    top_k = st.slider("Top-K Evidence", 3, 12, 6)  # Top-K slider
    search_mode = st.radio("Retrieval", ["Hybrid", "Similarity", "MMR - diverse evidence"], index=0)  # Search mode
    
    st.divider()  # Divider
    st.markdown("#### 🔑 API Key")  # API key section
    groq_key = st.text_input("GROQ_API_KEY", type="password", value=os.getenv("GROQ_API_KEY",""))  # Key input
    if groq_key:  # If key provided
        os.environ["GROQ_API_KEY"] = groq_key  # Set env variable

    indexed_meta = load_index_meta()  # Load meta for display
    if indexed_meta:  # If books indexed
        st.markdown(f"**Library:** {len(indexed_meta)} books")  # Show count
        for info in list(indexed_meta.values())[-5:]:  # Show last 5
            st.caption(f"• {info['book_title']} ({info['chunks']} chunks)")  # Show book

# Main hero section
st.markdown("""
<div class="hero">
  <div class="hero-kicker">ISLAMIC KNOWLEDGE • ROMAN URDU • QURAN DETECTOR • AUDIO</div>
  <h1>Kamran Islamic RAG Chatbot — Ask in English, Urdu, Arabic or Roman Urdu</h1>
  <p>Upload any Islamic book as text or scanned image. Auto OCR, Roman Urdu to Urdu search, Quran ayah auto-linking to quran.com, and audio playback for Arabic. Fast persistent indexing.</p>
</div>
""", unsafe_allow_html=True)

# Initialize vectorstore
vectorstore = None  # Default None
if not uploaded_files and not indexed_meta:  # First run no files
    c1, c2, c3, c4 = st.columns(4)  # 4 metric cards (original pattern)
    with c1:
        st.markdown('<div class="glass-card"><div class="card-label">Input Types</div><div class="card-value">Image + Text</div><div class="card-sub">Auto OCR ara+urd+eng</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><div class="card-label">Roman Urdu</div><div class="card-value">Enabled</div><div class="card-sub">wazu → وضو search</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass-card"><div class="card-label">Quran Detector</div><div class="card-value">Auto Link</div><div class="card-sub">2:255 → quran.com</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="glass-card"><div class="card-label">Audio</div><div class="card-value">TTS</div><div class="card-sub">Arabic recitation</div></div>', unsafe_allow_html=True)
    st.info("👈 Upload PDFs from sidebar. Library persists on disk.")

    if any(PERSIST_DIR.iterdir()):  # Try load existing
        try:
            embedding = load_embedding_model()  # Load embedding
            vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embedding, collection_name=COLLECTION_NAME)  # Load Chroma
            st.success(f"Loaded existing library: {len(indexed_meta)} books")  # Show success
        except Exception as e:
            st.error(f"Could not load: {e}")  # Show error
else:
    if uploaded_files:  # If new files uploaded
        with st.status("📖 Indexing Islamic Books...", expanded=True) as status:  # Status container
            prog = st.progress(0, text="Starting...")  # Progress bar
            vectorstore, pages, chunks, skipped, new_books, meta = build_knowledge_base(uploaded_files, chunk_size, overlap, prog)  # Build KB
            status.update(label=f"✅ Library Ready — {len(meta)} books total", state="complete")  # Update status
        if new_books:  # Show newly indexed
            st.success(f"Newly indexed: {', '.join(new_books)} — {chunks} new chunks")
        if skipped:  # Show skipped
            st.info(f"Skipped (already indexed): {', '.join(skipped)}")
    else:  # Load existing without new uploads
        embedding = load_embedding_model()  # Load embedding
        vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embedding, collection_name=COLLECTION_NAME)  # Load

# Main app tabs if vectorstore ready
if vectorstore:
    try:
        count = vectorstore._collection.count()  # Get chunk count
    except:
        count = "—"  # Fallback
    
    m1, m2, m3, m4 = st.columns(4)  # Stats row
    with m1:
        st.markdown(f'<div class="glass-card"><div class="card-label">Total Books</div><div class="card-value">{len(load_index_meta())}</div><div class="card-sub">Persistent</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><div class="card-label">Total Chunks</div><div class="card-value">{count}</div><div class="card-sub">Chunk {chunk_size}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="glass-card"><div class="card-label">OCR</div><div class="card-value">{"Enabled" if HAS_OCR else "Text Only"}</div><div class="card-sub">Roman Urdu Map</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="glass-card"><div class="card-label">Mode</div><div class="card-value">{search_mode}</div><div class="card-sub">Top-K {top_k}</div></div>', unsafe_allow_html=True)

    st.divider()  # Divider
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Library", "📚 Library Manager", "🔍 Evidence Explorer", "⚡ Lab & Quran"])  # Tabs

    with tab1:  # Ask tab
        st.markdown("### Ask anything - Supports Roman Urdu!")  # Title
        st.caption("Try: 'wazu ke faraiz kya hain?' or 'namaz ka tarika' or 'What is niyyah?' - All will search Urdu/Arabic books")  # Helper

        query = st.text_input("Your question", placeholder="e.g., wazu ke faraiz kya hain? / وضو کے فرائض / What are conditions of Salah? / 2:255 ki tafsir")  # Query input
        
        col_ask, col_clear = st.columns([1,0.2])  # Buttons
        with col_ask:
            ask_btn = st.button("Ask Noor Library", type="primary", use_container_width=True)  # Ask button
        with col_clear:
            if st.button("Clear Index", use_container_width=True):  # Clear button
                import shutil  # Import shutil
                shutil.rmtree(PERSIST_DIR, ignore_errors=True)  # Delete Chroma dir
                PERSIST_DIR.mkdir(exist_ok=True)  # Recreate
                st.rerun()  # Rerun app

        if ask_btn and query.strip():  # If ask clicked and query not empty
            if not os.getenv("GROQ_API_KEY"):  # Check API key
                st.error("Please add GROQ_API_KEY in sidebar or .env")  # Error
            else:
                # Show Roman Urdu expansion
                expanded = roman_urdu_to_urdu(query)  # Expand query
                if expanded != query.lower():  # If expansion happened
                    st.info(f"🔍 Roman Urdu detected → Searching for: `{expanded}`")  # Show expansion

                # Detect Quran refs in query
                quran_refs = detect_quran_refs(query)  # Detect refs
                if quran_refs:  # If found
                    for ref in quran_refs:  # Show each
                        st.markdown(f'<span class="quran-ref">📖 Quran {ref["surah"]}:{ref["ayah"]} → <a href="{ref["url"]}" target="_blank">quran.com</a></span>', unsafe_allow_html=True)

                with st.spinner("Searching your Islamic books..."):  # Spinner
                    docs = retrieve_documents(vectorstore, query, search_mode, top_k)  # Retrieve docs
                    context = create_context(docs)  # Create context

                if not docs:  # No docs found
                    st.warning("No relevant passages found. Try rephrasing or upload more books.")  # Warning
                else:
                    llm = get_llm()  # Get LLM
                    prompt = f"""You are Noor Library, an Islamic knowledge assistant. Answer ONLY from provided book excerpts.
Rules:
- Use only context. If not in context, say: "This is not found in uploaded books - یە کتابوں میں موجود نہیں"
- Always cite: [Book Name, Page X] after each fact.
- If context contains Quran/Hadith, quote Arabic then translation.
- Preserve respect: ﷺ for Prophet, رضي الله عنه for Sahaba.
- Answer in language of question (English/Urdu/Arabic/Roman Urdu). For Roman Urdu question, answer in Roman Urdu + Urdu.
- No fatwa - provide book evidence, suggest consulting scholar.
- If query is Roman Urdu, try to answer in same Roman Urdu + include Urdu script.

QUESTION: {query}
EXPANDED SEARCH TERMS: {expanded}

CONTEXT FROM ISLAMIC BOOKS:
{context}

Provide answer with citations.
"""  # Prompt with strict Islamic rules
                    with st.spinner("Generating cited answer..."):  # Spinner
                        answer = invoke_with_fallback(prompt).content  # Generate answer

                    st.markdown("#### 📖 Answer from Your Library")  # Answer header
                    st.success(answer)  # Show answer

                    # Detect Quran refs in answer for auto-linking
                    ans_refs = detect_quran_refs(answer)  # Detect in answer
                    if ans_refs:  # If found
                        st.markdown("#### 🔗 Quran References Detected")  # Header
                        for ref in ans_refs:  # Loop
                            st.markdown(f"- **{ref['text']}** → [Open in Quran.com]({ref['url']})")  # Link

                    # Audio playback for Arabic portions
                    arabic_parts = re.findall(r'[\u0600-\u06FF\s]{20,}', answer)  # Find Arabic text
                    if arabic_parts and HAS_TTS:  # If Arabic and TTS available
                        with st.expander("🔊 Audio Playback - Arabic portions"):  # Expander
                            for idx, arabic_text in enumerate(arabic_parts[:2]):  # First 2 Arabic blocks
                                st.markdown(f"**Arabic {idx+1}:** {arabic_text[:200]}...")  # Show text
                                if st.button(f"Play Audio {idx+1}", key=f"audio_{idx}"):  # Play button
                                    audio_path = generate_audio(arabic_text, lang='ar')  # Generate audio
                                    if audio_path:  # If generated
                                        st.audio(audio_path)  # Play audio

                    st.markdown("#### 📑 Evidence Used")  # Evidence header
                    for i, d in enumerate(docs, 1):  # Loop docs
                        src = d.metadata.get("book_title", d.metadata.get("source_file", "Unknown"))  # Source
                        page = d.metadata.get("page", 0)+1  # Page
                        ext = d.metadata.get("extraction", "")  # Extraction type
                        is_arabic = any('\u0600' <= c <= '\u06FF' for c in d.page_content[:100])  # Check Arabic
                        st.markdown(f"""
                        <div class="evidence-card">
                            <span class="source-chip">{src}</span>
                            <span class="source-chip">Page {page}</span>
                            <span class="source-chip">{ext}</span>
                            <div style="margin-top:.6rem" class="{'arabic-text' if is_arabic else ''}">{d.page_content[:1000]}</div>
                        </div>
                        """, unsafe_allow_html=True)  # Show evidence card

        st.markdown("---")  # Divider
        st.markdown("##### 💡 Try Roman Urdu Examples")  # Examples
        ex1, ex2, ex3 = st.columns(3)  # 3 columns
        with ex1:
            st.code("wazu ke faraiz kya hain?")  # Example 1
        with ex2:
            st.code("namaz ka tarika batao")  # Example 2
        with ex3:
            st.code("roza ki fazilat hadith me")  # Example 3

    with tab2:  # Library Manager
        st.markdown("### Library Manager")  # Title
        meta = load_index_meta()  # Load meta
        if not meta:  # No books
            st.info("No books indexed yet.")  # Info
        else:
            for h, info in meta.items():  # Loop books
                with st.expander(f"📘 {info['book_title']} — {info['chunks']} chunks, {info['pages']} pages"):  # Expander
                    st.json(info)  # Show JSON

    with tab3:  # Evidence Explorer (original lab pattern)
        st.markdown("### Raw Evidence Explorer - No LLM")  # Title
        q2 = st.text_input("Search evidence directly", key="ev_search", placeholder="e.g., wazu, niyyah, صلاة, namaz")  # Search
        if q2:  # If query
            expanded2 = roman_urdu_to_urdu(q2)  # Expand Roman Urdu
            st.caption(f"Searching: {expanded2}")  # Show expanded
            docs = retrieve_documents(vectorstore, q2, search_mode, top_k=10)  # Retrieve
            for i, d in enumerate(docs, 1):  # Loop
                src = d.metadata.get("book_title", "Unknown")  # Source
                page = d.metadata.get("page", 0)+1  # Page
                with st.expander(f"{i}. {src} | Page {page} | {d.metadata.get('extraction')}"):  # Expander
                    st.write(d.page_content)  # Show content

    with tab4:  # Performance Lab + Quran (original roadmap expanded)
        st.markdown("### Performance Optimizations & Features")  # Title
        st.markdown("""
        **From Original Aicademy 360 RAG:**
        - `@st.cache_resource` for embeddings & LLM (load once)
        - `RecursiveCharacterTextSplitter` with chunk_size + overlap sliders
        - `similarity_search` vs `max_marginal_relevance_search` (MMR)
        - Temp file pattern with `os.unlink` cleanup
        - Source metadata + page tracking

        **New Islamic Optimizations:**
        - **Persistent ChromaDB**: Saved to disk, no re-embedding on restart
        - **Hash Deduplication**: MD5 check, skip already indexed books
        - **Hybrid OCR Pipeline**: PyMuPDF fast path → only OCR if <80 chars
        - **Batch Embeddings**: batch_size=64, normalized
        - **Roman Urdu Map**: 30+ mappings wazu→وضو, namaz→نماز etc. + auto-expansion
        - **Quran Detector**: Regex 2:255 → auto links to quran.com/{surah}/{ayah}
        - **Audio TTS**: gTTS for Arabic recitation playback
        - **Arabic Normalization**: Tatweel + Alef normalization for better recall
        - **Multilingual Model**: paraphrase-multilingual-MiniLM-L12-v2 (80MB, Arabic/Urdu/English)
        """)  # Show features

        if st.button("Show Disk Usage"):  # Disk usage button
            import shutil  # Import
            size = sum(f.stat().st_size for f in PERSIST_DIR.rglob('*') if f.is_file()) / (1024*1024)  # Calculate size
            st.metric("ChromaDB Size", f"{size:.2f} MB")  # Show metric

        st.markdown("#### 🧪 Test Quran Detector")  # Test section
        test_text = st.text_input("Test Quran ref detection", placeholder="e.g., Surah Baqarah 2:255 is Ayatul Kursi")  # Test input
        if test_text:  # If input
            refs = detect_quran_refs(test_text)  # Detect
            if refs:  # If found
                for ref in refs:
                    st.success(f"Detected: {ref['text']} → {ref['url']}")  # Show success
            else:
                st.warning("No Quran reference detected")  # Warning
