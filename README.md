# 🎬 Video QA System

A temporally-grounded Video Question Answering system that uses **OpenAI CLIP** for visual embeddings, **FAISS** for fast semantic retrieval, and **Gemini / OpenAI** for natural-language explanations. Upload a video, ask a question, and get back the exact timestamps where the answer appears.

---

## 🏗 Architecture

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
│  FAISS Index (Inner Product)    │  ← cosine similarity on L2-normalized vecs
│  → Cached to disk via MD5 hash  │
└────────────────┬────────────────┘
                 │
         ┌───────┘  (at query time)
         │
         ▼
┌─────────────────────────────────┐
│  Text Query → CLIP embedding    │
│  → FAISS search → top-K clips   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  LLM Explanation                │  ← Gemini Flash → OpenAI → Template fallback
└─────────────────────────────────┘
```

---

## 📦 Tech Stack

| Component         | Technology                      |
|-------------------|---------------------------------|
| Visual Embeddings | OpenAI CLIP ViT-B/32            |
| Vector Store      | FAISS (IndexFlatIP)             |
| Frame Extraction  | OpenCV                          |
| Temporal Grouping | Custom sliding-window grouping  |
| LLM Explanation   | Gemini 1.5 Flash / GPT-4o-mini  |
| Frontend          | Streamlit                       |

---

## 🚀 Getting Started

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd "TDL project"

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys (optional)

Copy the example env file and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_key_here      # Recommended — free tier available
OPENAI_API_KEY=your_openai_key_here      # Optional fallback
```

> **Note:** The system works **without any API key**. It falls back to a template-based explanation.

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🗂 Project Structure

```
TDL project/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
│
└── video_qa/               # Core pipeline package
    ├── __init__.py
    ├── embed.py            # CLIP image & text embedding (with safe error handling)
    ├── pipeline.py         # End-to-end orchestrator (ingest + query)
    ├── retrieval.py        # FAISS vector store (build / search / save / load)
    ├── temporal.py         # Frame → clip grouping & timestamp utilities
    ├── video_utils.py      # OpenCV frame extraction helpers
    └── llm.py              # LLM explanation (Gemini → OpenAI → template fallback)
```

---

## 🖥 Usage

### Ingest a Video

1. Open the **Ingest Video** tab
2. Upload a video file (MP4, AVI, MOV, MKV, WEBM — max 200 MB)
3. Click **⚡ Process Video**

The system will:
- Extract frames at 1 fps
- Compute CLIP embeddings for each frame
- Group frames into 2-second clips
- Build and cache a FAISS index (keyed by video MD5 hash)

### Ask a Question

1. Switch to the **❓ Ask a Question** tab
2. Type a natural-language question, e.g.:
   - *"When does the speaker mention gradient descent?"*
   - *"Show me the part where the balloon is released."*
3. Click **🔍 Search**

Results show ranked video clips with timestamps, relevance scores, and an AI-generated explanation.

---

## ⚙️ Configuration

The following parameters are currently fixed in `app.py` and can be adjusted:

| Parameter          | Default | Description                        |
|--------------------|---------|------------------------------------|
| `fps`              | `1.0`   | Frames extracted per second        |
| `clip_duration_sec`| `2.0`   | Seconds per temporal clip          |
| `top_k`            | `5`     | Number of clips returned per query |

---

## 🔑 Key Design Choices

- **Clip-level retrieval** (not frame-level): Frames are grouped into 2-second windows and their embeddings are averaged → provides true temporal context and reduces index size.
- **Cosine similarity**: All vectors are L2-normalized before indexing; FAISS inner-product then equals cosine similarity — no extra normalization at query time.
- **Disk caching**: The FAISS index and metadata are saved under `.vqa_cache/<video_md5>/`. Re-uploading the same video loads from cache instantly.
- **Graceful LLM fallback**: Gemini is tried first, then OpenAI, then a built-in template — the pipeline never crashes due to a missing API key.
- **Safe frame loading**: Corrupt or missing frame files are skipped with a warning rather than crashing the embedding step.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `PytorchStreamReader failed reading zip archive` | Delete corrupted CLIP cache: `rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` and restart |
| `No module named 'faiss'` | Install with `pip install faiss-cpu` |
| `cv2` import error | Install `pip install opencv-python-headless` |
| App slow on first run | CLIP model (~600 MB) downloads on first use — subsequent runs are fast |
| Gemini quota exceeded | Add `OPENAI_API_KEY` as fallback, or leave blank for template explanations |

---

## 📄 License

This project is built for academic/educational purposes as part of a TDL (Theory of Deep Learning) course project.
