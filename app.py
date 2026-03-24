"""
app.py  —  Streamlit frontend for the Advanced Video QA System

Run with:
    streamlit run app.py
"""

import logging
import os
import sys
import tempfile

import streamlit as st
from PIL import Image

# ── Ensure the project root is on sys.path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from video_qa.pipeline import VideoQAPipeline
from video_qa.temporal import format_timestamp

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Video QA System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}

/* Main title */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Cards */
.clip-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: transform 0.2s;
}
.clip-card:hover {
    transform: translateY(-2px);
    border-color: rgba(167,139,250,0.5);
}

/* Timestamp badge */
.ts-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Score badge */
.score-badge {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid #34d399;
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-left: 0.5rem;
}

/* Explanation box */
.explanation-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(37,99,235,0.15));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    color: #e2e8f0;
    line-height: 1.7;
}
.explanation-label {
    font-size: 0.8rem;
    color: #a78bfa;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #34d399) !important;
}

/* Info boxes */
.info-pill {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    color: #93c5fd;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.82rem;
    margin: 0.2rem;
}

/* Dividers */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Session state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_pipeline() -> VideoQAPipeline:
    """Return (or create) the pipeline stored in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    return st.session_state.pipeline


def _reset_pipeline(fps, clip_dur):
    """Create a fresh pipeline in session state."""
    st.session_state.pipeline = VideoQAPipeline(
        fps=fps,
        clip_duration_sec=clip_dur,
        frames_folder=os.path.join(tempfile.gettempdir(), "vqa_frames"),
    )
    st.session_state.ingest_summary = None
    st.session_state.query_result = None


def _compute_playback_window(
    clip_start: float,
    clip_end: float,
    video_duration: float | None,
    pre_sec: float = 6.0,
    post_sec: float = 8.0,
    min_window_sec: float = 8.0,
) -> tuple[float, float]:
    """
    Expand a retrieved clip into a wider incident window for easier validation.
    """
    start = max(0.0, float(clip_start) - pre_sec)
    end = max(start + 0.5, float(clip_end) + post_sec)

    # Ensure a minimum context window (avoid tiny 1-second playback)
    if end - start < min_window_sec:
        pad = (min_window_sec - (end - start)) / 2.0
        start = max(0.0, start - pad)
        end = end + pad

    if video_duration is not None:
        end = min(float(video_duration), end)
        if end - start < min_window_sec:
            start = max(0.0, end - min_window_sec)

    return start, end


# ══════════════════════════════════════════════════════════════════════════════
#  Fixed Parameters
# ══════════════════════════════════════════════════════════════════════════════
fps = 1.0
clip_dur = 2.0
top_k = 5

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Video QA System**  
    Uses CLIP + FAISS for temporal clip retrieval.  
    Supports Gemini & OpenAI for explanations.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">🎬 Video QA System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a video · Ask a question · Get temporally-grounded answers</div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_query, tab_about = st.tabs(["📥 Ingest Video", "❓ Ask a Question", "📚 How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Ingest
# ══════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.markdown("### Upload your video")
    uploaded = st.file_uploader(
        "Drop a video file here",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )

    col_btn, col_force = st.columns([3, 1])
    with col_btn:
        process_btn = st.button("⚡ Process Video", use_container_width=True)
    with col_force:
        force_reprocess = st.checkbox("Force re-process", value=False,
                                      help="Ignore cached index and re-embed everything.")

    if process_btn:
        if not uploaded:
            st.error("Please upload a video file first.")
        else:
            # Save uploaded file to temp
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            # Reset pipeline with current settings
            _reset_pipeline(fps, clip_dur)
            pipeline = st.session_state.pipeline

            # Progress UI
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def on_progress(step: str, pct: float):
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(f"**{step}**")

            with st.spinner(""):
                try:
                    summary = pipeline.ingest(
                        tmp_path,
                        force=force_reprocess,
                        progress_callback=on_progress,
                    )
                    st.session_state.ingest_summary = summary
                    progress_bar.progress(1.0)
                    status_text.empty()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    st.stop()

    # Show summary if available
    if st.session_state.get("ingest_summary"):
        s = st.session_state.ingest_summary
        st.success("✅ Video processed and ready to query!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📹 Duration", f"{s['duration_sec']:.1f}s")
        col2.metric("🖼 Frames", str(s["num_frames"]))
        col3.metric("🎞 Clips", str(s["num_clips"]))
        col4.metric("💾 Source", "Cache" if s.get("cached") else "Fresh")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Query
# ══════════════════════════════════════════════════════════════════════════════

with tab_query:
    pipeline = _get_pipeline()

    if pipeline is None or not pipeline.is_ready:
        st.info("📥 Please process a video first in the **Ingest Video** tab.")
    else:
        st.markdown("### Ask a question about the video")

        with st.form("query_form"):
            question = st.text_input(
                "Your question",
                placeholder='e.g. "When does the speaker mention gradient descent?"',
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

        if submitted and question.strip():
            with st.spinner("Searching video and generating explanation …"):
                try:
                    result = pipeline.query(question.strip(), k=top_k)
                    st.session_state.query_result = result
                    st.session_state.last_question = question.strip()
                except Exception as e:
                    st.error(f"Query failed: {e}")

        # ── Results ──────────────────────────────────────────────────────────
        if st.session_state.get("query_result"):
            result = st.session_state.query_result
            q_text = st.session_state.get("last_question", "")
            clips = result["clips"]
            explanation = result["explanation"]
            source_video = pipeline.video_path if pipeline else None
            video_duration = None
            if st.session_state.get("ingest_summary"):
                video_duration = st.session_state.ingest_summary.get("duration_sec")

            st.markdown(f"---\n### 🎯 Results for: *\"{q_text}\"*")
            st.markdown(f"Found **{len(clips)} relevant clips**")

            # Clip cards
            for rank, clip in enumerate(clips, 1):
                ts_start = format_timestamp(clip["start_sec"])
                ts_end = format_timestamp(clip["end_sec"])
                score_pct = min(clip.get("score", 0) * 100, 100)
                rep_frame = clip.get("representative_frame", "")
                play_start, play_end = _compute_playback_window(
                    clip.get("start_sec", 0.0),
                    clip.get("end_sec", 0.0),
                    video_duration,
                )

                col_img, col_info = st.columns([1, 2])

                with col_img:
                    if source_video and os.path.exists(source_video):
                        try:
                            st.video(source_video, start_time=play_start, end_time=play_end)
                        except TypeError:
                            # Older Streamlit versions may not support end_time
                            st.video(source_video, start_time=play_start)
                    elif rep_frame and os.path.exists(rep_frame):
                        img = Image.open(rep_frame).convert("RGB")
                        st.image(img, use_container_width=True)
                    else:
                        st.markdown("*Playback/frame not available*")

                with col_info:
                    st.markdown(
                        f'<span class="ts-badge">🕐 {ts_start} → {ts_end}</span>'
                        f'<span class="score-badge">score {score_pct:.1f}%</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Rank #{rank}** &nbsp;|&nbsp; Clip {clip['clip_id']}"
                                f" &nbsp;|&nbsp; {clip.get('num_frames', '?')} frames sampled")
                    st.markdown(
                        f"Playback window: **{format_timestamp(play_start)} → {format_timestamp(play_end)}**"
                    )

                    # Mini score bar
                    st.progress(score_pct / 100, text=f"Relevance: {score_pct:.1f}%")

                st.markdown('<hr style="margin:0.5rem 0"/>', unsafe_allow_html=True)

            # LLM Explanation
            st.markdown(
                f'<div class="explanation-box">'
                f'<div class="explanation-label">🤖 AI Explanation</div>'
                f'{explanation}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: About
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown("""
### 🏗 System Architecture

This system implements a full **Temporal Video Question Answering pipeline**:

```
Video File
    │
    ▼
┌─────────────────────────────────┐
│  Frame Extraction (OpenCV)      │  ← 1 frame/sec (configurable)
│  → timestamp_sec per frame      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  CLIP Visual Embeddings         │  ← openai/clip-vit-base-patch32
│  → 512-dim L2-normalized vector │
│     per frame                   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Temporal Clip Grouping         │  ← 2-second windows (configurable)
│  → Average embedding per clip   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  FAISS Index (Inner Product)    │  ← cosine similarity on normalized vecs
│  → Cached to disk via MD5 hash  │
└────────────────┬────────────────┘
                 │
         ┌───────┘ (at query time)
         │
         ▼
┌─────────────────────────────────┐
│  Text Query → CLIP embedding    │
│  → FAISS search → top-K clips   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  LLM Explanation                │  ← Gemini Flash → OpenAI → Template
└─────────────────────────────────┘
```

### 📦 Tech Stack

| Component | Technology |
|---|---|
| Visual Embeddings | OpenAI CLIP ViT-B/32 |
| Vector Store | FAISS (IndexFlatIP) |
| Frame Extraction | OpenCV |
| Temporal Grouping | Custom sliding window |
| LLM | Gemini 1.5 Flash / GPT-4o-mini |
| Frontend | Streamlit |

### 💡 Key Design Choices

- **Clip-level retrieval** (not frame-level): frames are grouped into 2s clips,
  embeddings are averaged → true temporal context
- **Cosine similarity**: all vectors are L2-normalized, FAISS inner-product ≡ cosine sim
- **Disk caching**: FAISS index + metadata saved by video MD5 hash — fast reload
- **Graceful LLM fallback**: works without any API key (template explanation)
""")
