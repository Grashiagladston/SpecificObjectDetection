import os
import io
from datetime import datetime
import streamlit as st
from PIL import Image
import uuid

# Imports
import config
from database import (
    init_db, 
    save_embeddings_only,  # NEW: Only saves ID + embeddings
    get_history_records, 
    clear_all_history, 
    find_similar_images
)
from gemini_detector import GeminiDetector
from draw_boxes import draw_annotations
from crop_utils import crop_detected_objects
from embedder import generate_embedding

# --- Setup ---
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    init_db()
except Exception as e:
    st.error(f"Database Connection Failed. Check `config.py`. Error: {e}")
    st.stop()

detector = GeminiDetector()

# --- Comprehensive Styling (Same as before) ---
st.markdown("""
<style>
:root {
    --primary: #00F0FF;
    --primary-dim: rgba(0, 240, 255, 0.15);
    --primary-glow: rgba(0, 240, 255, 0.4);
    --accent: #7B61FF;
    --accent-dim: rgba(123, 97, 255, 0.15);
    --success: #00E676;
    --warning: #FFD740;
    --danger: #FF5252;
    --bg-dark: #0A0E1A;
    --bg-card: #111827;
    --bg-card-hover: #1A2234;
    --bg-surface: #151C2C;
    --border: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(0, 240, 255, 0.2);
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow-glow: 0 0 20px rgba(0, 240, 255, 0.15);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stApp {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { 
    background: rgba(10, 14, 26, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid var(--border) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--primary-dim); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-glow); }

.header-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.header-container::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
}
.title-gradient {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00F0FF 0%, #7B61FF 50%, #00F0FF 100%);
    background-size: 200% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 4s ease-in-out infinite;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem !important;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}
.subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.03em;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-dark) 100%) !important;
    border-right: 1px solid var(--border-accent) !important;
}
section[data-testid="stSidebar"] .stTitle {
    color: var(--primary) !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
}
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    padding: 0.6rem 0.8rem !important;
    font-size: 0.9rem !important;
    transition: var(--transition);
}
section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-dim) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    transition: var(--transition);
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(123,97,255,0.15)) !important;
    color: var(--primary) !important;
    box-shadow: var(--shadow-glow) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--primary) !important;
    height: 2px !important;
}
.stTabs [data-baseweb="tab-content"] {
    padding-top: 1.5rem !important;
}

.stFileUploader > div > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
    padding: 2rem !important;
    transition: var(--transition);
}
.stFileUploader > div > div:hover {
    border-color: var(--primary) !important;
    background: var(--bg-card-hover) !important;
    box-shadow: var(--shadow-glow) !important;
}
.stFileUploader > div > div label { color: var(--text-secondary) !important; }
.stFileUploader > div > div span { color: var(--text-muted) !important; font-size: 0.8rem !important; }

.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #00F0FF, #7B61FF) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #0A0E1A !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.02em;
    transition: var(--transition);
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 240, 255, 0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stButton > button[kind="secondary"] {
    background: rgba(255, 82, 82, 0.15) !important;
    color: var(--danger) !important;
    border: 1px solid rgba(255, 82, 82, 0.3) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255, 82, 82, 0.25) !important;
    box-shadow: 0 4px 15px rgba(255, 82, 82, 0.2) !important;
}

.result-image-container {
    border: 1px solid var(--border-accent);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--bg-card);
    box-shadow: var(--shadow-card);
}
.result-image-label {
    padding: 0.75rem 1rem;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
}

.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: var(--transition);
}
.streamlit-expanderHeader:hover {
    border-color: var(--border-accent) !important;
    background: var(--bg-card-hover) !important;
}
[data-testid="stExpander"] details { border: none !important; background: transparent !important; }
[data-testid="stExpander"] details[open] .streamlit-expanderHeader {
    border-radius: var(--radius) var(--radius) 0 0 !important;
    border-bottom: 1px solid var(--border-accent) !important;
}
[data-testid="stExpander"] details[open] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 1.2rem !important;
}

.stAlert {
    border-radius: var(--radius) !important;
    border: none !important;
    padding: 1rem 1.2rem !important;
    font-weight: 500 !important;
}
.stAlert[data-baseweb="notification"][kind="success"] {
    background: rgba(0, 230, 118, 0.1) !important;
    border-left: 3px solid var(--success) !important;
    color: var(--success) !important;
}
.stAlert[data-baseweb="notification"][kind="warning"] {
    background: rgba(255, 215, 64, 0.1) !important;
    border-left: 3px solid var(--warning) !important;
    color: var(--warning) !important;
}
.stAlert[data-baseweb="notification"][kind="error"] {
    background: rgba(255, 82, 82, 0.1) !important;
    border-left: 3px solid var(--danger) !important;
    color: var(--danger) !important;
}
.stAlert[data-baseweb="notification"][kind="info"] {
    background: rgba(0, 240, 255, 0.08) !important;
    border-left: 3px solid var(--primary) !important;
    color: var(--primary) !important;
}

.stCodeBlock { background: var(--bg-dark) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stCode"] { background: var(--bg-dark) !important; border-radius: var(--radius-sm) !important; }

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
    margin: 1.5rem 0;
}

.info-badge {
    display: inline-block;
    background: var(--primary-dim);
    color: var(--primary);
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.info-badge.accent { background: var(--accent-dim); color: var(--accent); }
.info-badge.danger { background: rgba(255, 82, 82, 0.15); color: var(--danger); }

.stSpinner > div { border-top-color: var(--primary) !important; }

.target-display {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, var(--primary-dim), var(--accent-dim));
    border: 1px solid var(--border-accent);
    border-radius: var(--radius);
    padding: 0.5rem 1rem;
    margin: 0.5rem 0 1rem;
}
.target-display .target-label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
.target-display .target-value { color: var(--primary); font-weight: 700; font-size: 1rem; font-family: 'JetBrains Mono', monospace; }

.embedding-preview {
    background: var(--bg-dark);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    word-break: break-all;
    line-height: 1.6;
}
.embedding-preview .dim-info { color: var(--accent); font-weight: 600; }

.empty-state { text-align: center; padding: 3rem 1rem; color: var(--text-muted); }
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-state p { font-size: 0.9rem; }

[data-testid="stImage"] img {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-accent) !important;
}

/* Hide uploaded file thumbnail preview */
[data-testid="stFileUploader"] [data-testid="stImage"] {
    display: none !important;
}

/* Custom ID display styling */
.id-display {
    background: var(--bg-dark);
    border: 2px solid var(--border-accent);
    border-radius: var(--radius);
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.id-display .id-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.id-display .id-value {
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
}

/* Duplicate warning box */
.duplicate-warning-box {
    background: rgba(255, 82, 82, 0.08);
    border: 2px solid rgba(255, 82, 82, 0.3);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.duplicate-warning-box .warning-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.duplicate-warning-box .warning-title {
    color: var(--danger);
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.duplicate-warning-box .warning-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* New image success box */
.new-image-success-box {
    background: rgba(0, 230, 118, 0.08);
    border: 2px solid rgba(0, 230, 118, 0.3);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.new-image-success-box .success-icon {
    font-size: 2rem;
}
.new-image-success-box .success-content .success-title {
    color: var(--success);
    font-size: 1.1rem;
    font-weight: 700;
}
.new-image-success-box .success-content .success-subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* Crop grid */
.crop-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.crop-card {
    border: 1px solid var(--border-accent);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--bg-card);
    transition: var(--transition);
}
.crop-card:hover {
    border-color: var(--primary);
    box-shadow: var(--shadow-glow);
}
.crop-card img {
    width: 100%;
    display: block;
}
.crop-card .crop-label {
    padding: 0.6rem 0.8rem;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
}

/* Embedding section */
.embedding-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.embedding-section .section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.embedding-section .section-title {
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 1.6rem; margin-bottom: 0.3rem;">⚙️</div>
        <div style="color: var(--primary); font-weight: 700; font-size: 1.15rem; letter-spacing: 0.02em;">Settings</div>
        <div style="height: 2px; width: 40px; background: linear-gradient(90deg, var(--primary), var(--accent)); margin-top: 0.5rem; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    target_object = st.text_input("Target Object", value="cat", label_visibility="collapsed", placeholder="Enter object name...")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: var(--text-muted); font-size: 0.75rem; line-height: 1.6;">
        <div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 0.3rem;">How it works</div>
        1. Upload an image<br>
        2. Set the target object<br>
        3. Click <strong style="color: var(--primary);">Process Image</strong><br>
        4. <strong style="color: var(--success);">New:</strong> Shows crops + embeddings<br>
        5. <strong style="color: var(--danger);">Duplicate (≥85%):</strong> Only ID + embeddings<br>
        6. DB stores <strong>ONLY</strong> ID & embeddings
    </div>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div class='header-container'>
    <h1 class='title-gradient'>{config.PAGE_TITLE}</h1>
    <p class='subtitle'>Detection • Cropping • Embeddings • No Image Storage</p>
</div>
""", unsafe_allow_html=True)

tab_hub, tab_history = st.tabs(["🎯 Process Image", "📂 Embedding History"])

# --- Tab 1: Process Image ---
with tab_hub:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        st.markdown(f"""
        <div class="target-display">
            <span class="target-label">Target:</span>
            <span class="target-value">{target_object}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍  Process Image", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Step 1: Generate embedding for the uploaded image
                    uploaded_embedding = generate_embedding(image)
                    
                    # Step 2: Check for duplicates (85% threshold)
                    similar_runs = find_similar_images(uploaded_embedding, threshold=0.85)
                    
                    # =============================================
                    # DUPLICATE DETECTED (≥85% similarity)
                    # Return ONLY: Image ID + Embeddings
                    # NO IMAGES SHOWN
                    # =============================================
                    if similar_runs:
                        for (db_row, similarity) in similar_runs:
                            run_id, ts, name, target, count, db_emb = db_row
                            
                            # Warning box - NO IMAGE
                            st.markdown(f"""
                            <div class="duplicate-warning-box">
                                <div class="warning-icon">⚠️</div>
                                <div class="warning-title">DUPLICATE DETECTED</div>
                                <div class="warning-subtitle">Similarity: {similarity*100:.2f}% (Threshold: 85%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.error("🚫 Duplicate image detected. No image will be displayed as per policy.")
                            
                            # === IMAGE ID ONLY ===
                            st.markdown("""
                            <div class="id-display">
                                <div class="id-label">Matched Image ID</div>
                                <div class="id-value">""" + str(run_id) + """</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # === UPLOADED IMAGE EMBEDDING ===
                            up_dims = len(uploaded_embedding) if uploaded_embedding else 0
                            st.markdown(f"**📤 Uploaded Image Embedding** <span class='info-badge'>{up_dims} dims</span>", unsafe_allow_html=True)
                            
                            if uploaded_embedding:
                                up_preview = ", ".join(str(round(x, 4)) for x in uploaded_embedding[:10]) + ", ..."
                                st.code(f"[{up_preview}]", language="text")
                                
                                with st.expander("👁️ View Full Uploaded Vector"):
                                    st.code(str(list(uploaded_embedding)), language="json")
                            
                            st.markdown("---")
                            
                            # === STORED (MATCHED) EMBEDDING ONLY ===
                            if db_emb:
                                # Parse embedding from database
                                if isinstance(db_emb, str):
                                    try:
                                        db_emb_list = [float(x) for x in db_emb.strip("[]").split(",") if x.strip()]
                                    except:
                                        db_emb_list = [db_emb]
                                elif isinstance(db_emb, (list, tuple)):
                                    db_emb_list = list(db_emb)
                                else:
                                    db_emb_list = list(db_emb) if hasattr(db_emb, '__iter__') else [str(db_emb)]
                                
                                db_dims = len(db_emb_list)
                                st.markdown(f"**💾 Stored Embedding (ID: {run_id})** <span class='info-badge accent'>{db_dims} dims</span> <span class='info-badge danger'>DUPLICATE</span>", unsafe_allow_html=True)
                                
                                db_preview = ", ".join(str(round(x, 4)) for x in db_emb_list[:10]) + ", ..."
                                st.code(f"[{db_preview}]", language="text")
                                
                                with st.expander(f"👁️ View Full Stored Vector (ID: {run_id})"):
                                    st.code(str(db_emb_list), language="json")
                            else:
                                st.warning("No stored embedding found.")
                            
                            st.markdown("---")
                        
                        st.info("🔒 **Policy:** Only Image ID and Embeddings are returned. No images are stored or displayed for duplicates.")
                        st.stop()
                    
                    # =============================================
                    # NEW IMAGE (No duplicate found)
                    # SHOW: Cropped images + Embeddings
                    # STORE: ONLY ID + Embeddings (NO IMAGES)
                    # =============================================
                    
                    # Generate unique ID for this image
                    image_id = str(uuid.uuid4())[:8].upper()
                    
                    # Run detection
                    detections = detector.detect(image, target_object)
                    annotated_image, target_count, boxes_list = draw_annotations(image, detections, target_object)
                    crops_data = crop_detected_objects(image, detections, target_object)
                    
                    # Generate embeddings for each crop
                    crop_embeddings = []
                    for crop_item in crops_data:
                        crop_emb = generate_embedding(crop_item['image'])
                        crop_embeddings.append({
                            'label': crop_item['label'],
                            'embedding': crop_emb
                        })
                    
                    # SUCCESS BOX - NEW IMAGE
                    st.markdown(f"""
                    <div class="new-image-success-box">
                        <div class="success-icon">✅</div>
                        <div class="success-content">
                            <div class="success-title">NEW IMAGE PROCESSED</div>
                            <div class="success-subtitle">Found {target_count} object{"s" if target_count != 1 else ""} • Crops displayed below • Only ID & embeddings saved to DB</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === DISPLAY IMAGE ID ===
                    st.markdown(f"""
                    <div class="id-display">
                        <div class="id-label">Generated Image ID</div>
                        <div class="id-value">{image_id}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === DISPLAY CROPPED IMAGES (First Try - In Memory Only) ===
                    if crops_data:
                        st.markdown(f"""
                        <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
                            <span style="color: var(--text-secondary); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em;">
                                ✂️ Cropped Objects
                            </span>
                            <span class="info-badge">{len(crops_data)}</span>
                            <span class="info-badge accent">NOT SAVED TO DB</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        num_cols = min(len(crops_data), 4)
                        cols = st.columns(num_cols)
                        for idx, crop_item in enumerate(crops_data):
                            with cols[idx % num_cols]:
                                st.image(crop_item['image'], caption=crop_item['label'], use_container_width=True)
                    else:
                        st.info("No crops generated from this image.")
                    
                    # === DISPLAY FULL IMAGE EMBEDDING ===
                    st.markdown("---")
                    st.markdown("""
                    <div class="embedding-section">
                        <div class="section-header">
                            <span style="font-size: 1.2rem;">🔢</span>
                            <span class="section-title">Full Image Embedding</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if uploaded_embedding:
                        full_dims = len(uploaded_embedding)
                        st.markdown(f"<span class='info-badge'>{full_dims} dimensions</span>", unsafe_allow_html=True)
                        
                        full_preview = ", ".join(str(round(x, 4)) for x in uploaded_embedding[:10]) + ", ..."
                        st.code(f"[{full_preview}]", language="text")
                        
                        with st.expander("👁️ View Full Image Vector"):
                            st.code(str(list(uploaded_embedding)), language="json")
                    
                    # === DISPLAY CROP EMBEDDINGS ===
                    if crop_embeddings:
                        st.markdown("---")
                        st.markdown("""
                        <div class="embedding-section">
                            <div class="section-header">
                                <span style="font-size: 1.2rem;">📦</span>
                                <span class="section-title">Crop Embeddings</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for i, crop_emb_data in enumerate(crop_embeddings):
                            emb = crop_emb_data['embedding']
                            label = crop_emb_data['label']
                            dims = len(emb) if emb else 0
                            
                            st.markdown(f"**{label}** <span class='info-badge'>{dims} dims</span>", unsafe_allow_html=True)
                            
                            if emb:
                                crop_preview = ", ".join(str(round(x, 4)) for x in emb[:8]) + ", ..."
                                st.code(f"[{crop_preview}]", language="text")
                                
                                with st.expander(f"👁️ View Full Vector - {label}"):
                                    st.code(str(list(emb)), language="json")
                            
                            if i < len(crop_embeddings) - 1:
                                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    
                    # === SAVE TO DATABASE - ONLY ID + EMBEDDINGS ===
                    # NO IMAGE DATA IS STORED
                    save_embeddings_only(
                        image_id=image_id,
                        original_image_name=uploaded_file.name,
                        target_object=target_object,
                        detected_count=target_count,
                        full_image_embedding=uploaded_embedding,
                        crop_embeddings=crop_embeddings
                    )
                    
                    st.markdown("---")
                    st.success(f"✅ **Saved to Database:** Only Image ID (`{image_id}`) and embeddings. No image data stored.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- Tab 2: Embedding History (No Images) ---
with tab_history:
    history = get_history_records()
    
    if not history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p>No embedding history yet.<br>Upload an image to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        ">
            <div>
                <span class="info-badge">{len(history)} records</span>
                <span class="info-badge accent">Embeddings Only - No Images</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for row in history:
            run_id, ts, name, target, count, full_emb, crop_embs = row
            
            with st.expander(f"  🆔 {run_id}  |  {ts}  |  Target: {target}  |  Count: {count}"):
                
                # Show ID prominently
                st.markdown(f"""
                <div class="id-display">
                    <div class="id-label">Image ID</div>
                    <div class="id-value">{run_id}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Filename:** `{name}`")
                with col2:
                    st.markdown(f"**Target:** `{target}`")
                with col3:
                    st.markdown(f"**Objects Found:** `{count}`")
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                # Show Full Image Embedding
                if full_emb:
                    emb_preview = ", ".join(f"{x:.4f}" for x in full_emb[:8]) + ", ..."
                    st.markdown(f"""
                    <div style="margin-bottom: 0.75rem;">
                        <span class="info-badge">Full Image Embedding</span>
                        <span class="info-badge accent">{len(full_emb)} dimensions</span>
                    </div>
                    <div class="embedding-preview">
                        <span class="dim-info">Vector:</span> [{emb_preview}]
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("👁️ View Full Image Vector"):
                        st.code(str(full_emb), language="json")
                
                # Show Crop Embeddings (if any)
                if crop_embs:
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    st.markdown("**📦 Crop Embeddings:**")
                    
                    for crop_emb_data in crop_embs:
                        if isinstance(crop_emb_data, dict):
                            label = crop_emb_data.get('label', 'Unknown')
                            emb = crop_emb_data.get('embedding', [])
                        else:
                            label = 'Unknown'
                            emb = crop_emb_data if isinstance(crop_emb_data, list) else []
                        
                        if emb:
                            crop_emb_preview = ", ".join(f"{x:.4f}" for x in emb[:6]) + "..."
                            st.markdown(f"• **{label}** <span class='info-badge'>{len(emb)} dims</span>", unsafe_allow_html=True)
                            st.code(f"[{crop_emb_preview}]", language="text")
                            
                            with st.expander(f"View Full Vector - {label}"):
                                st.code(str(emb), language="json")
                
                # No image warning
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                st.info("🔒 No image data stored. Only ID and embeddings are available.")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        if st.button("🗑️  Clear All History", use_container_width=True):
            clear_all_history()
            st.rerun()