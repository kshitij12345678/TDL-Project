"""
pipeline.py
End-to-end orchestrator for the Video QA system.
Handles ingestion (extract → embed → group → index) and query (embed text → search → explain).
Caches the FAISS index to disk to avoid re-processing the same video.
"""

import hashlib
import logging
import os
import pickle

import numpy as np

from video_qa.embed import embed_images, embed_text
from video_qa.llm import explain
from video_qa.retrieval import VideoRetriever
from video_qa.temporal import compute_clip_embeddings, group_into_clips
from video_qa.video_utils import extract_frames, get_video_duration

logger = logging.getLogger(__name__)

CACHE_DIR = ".vqa_cache"
EMBEDDING_DIM = 512  # CLIP ViT-B/32


def _video_hash(video_path: str) -> str:
    """Compute a short hash of the video file for cache keying."""
    h = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class VideoQAPipeline:
    """
    Full Video QA pipeline.

    Example:
        pipeline = VideoQAPipeline(fps=1, clip_duration_sec=2.0)
        pipeline.ingest("lecture.mp4")
        result = pipeline.query("When does the presenter show the graph?", k=5)
        print(result["explanation"])
        for clip in result["clips"]:
            print(clip["start_sec"], clip["representative_frame"])
    """

    def __init__(
        self,
        fps: float = 1.0,
        clip_duration_sec: float = 2.0,
        frames_folder: str = "frames",
        cache_dir: str = CACHE_DIR,
    ):
        self.fps = fps
        self.clip_duration_sec = clip_duration_sec
        self.frames_folder = frames_folder
        self.cache_dir = cache_dir

        self.retriever = VideoRetriever(dim=EMBEDDING_DIM)
        self.frame_infos: list[dict] = []
        self.clips: list[dict] = []
        self.video_path: str | None = None
        self._ingested = False

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(
        self,
        video_path: str,
        force: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Process a video: extract frames, compute embeddings, build FAISS index.

        Args:
            video_path:        Path to the .mp4 (or other) video file.
            force:             If True, ignore cached index and re-process.
            progress_callback: Optional callable(step: str, pct: float) for UI updates.

        Returns:
            Summary dict with 'num_frames', 'num_clips', 'duration_sec'.
        """
        video_path = os.path.abspath(video_path)
        self.video_path = video_path

        def _report(step, pct):
            if progress_callback:
                progress_callback(step, pct)
            logger.info(f"[{int(pct * 100):3d}%] {step}")

        # ── Check cache ──
        vid_hash = _video_hash(video_path)
        cache_folder = os.path.join(self.cache_dir, vid_hash)
        meta_path = os.path.join(cache_folder, "ingest_meta.pkl")

        if not force and os.path.exists(meta_path):
            _report("Loading cached index …", 0.0)
            self.retriever.load(cache_folder)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.frame_infos = meta["frame_infos"]
            self.clips = meta["clips"]
            self._ingested = True
            _report("Cache loaded!", 1.0)
            logger.info(f"Loaded from cache ({len(self.clips)} clips)")
            return {
                "num_frames": len(self.frame_infos),
                "num_clips": len(self.clips),
                "duration_sec": get_video_duration(video_path),
                "cached": True,
            }

        # ── Step 1: Extract frames ──
        _report("Extracting frames …", 0.05)
        self.frame_infos = extract_frames(
            video_path,
            output_folder=self.frames_folder,
            fps=self.fps,
        )
        _report(f"Extracted {len(self.frame_infos)} frames", 0.25)

        # ── Step 2: Embed frames ──
        _report("Computing CLIP embeddings …", 0.30)
        frame_embeddings = embed_images(self.frame_infos)
        _report("Embeddings done", 0.65)

        # ── Step 3: Group into clips ──
        _report("Grouping frames into temporal clips …", 0.70)
        self.clips = group_into_clips(self.frame_infos, self.clip_duration_sec)
        clip_embeddings = compute_clip_embeddings(self.clips, frame_embeddings)
        _report(f"Grouped into {len(self.clips)} clips", 0.80)

        # ── Step 4: Build FAISS index ──
        _report("Building FAISS index …", 0.85)
        self.retriever.build_index(clip_embeddings, self.clips)

        # ── Step 5: Save cache ──
        _report("Saving index cache …", 0.95)
        self.retriever.save(cache_folder)
        with open(meta_path, "wb") as f:
            pickle.dump({"frame_infos": self.frame_infos, "clips": self.clips}, f)

        self._ingested = True
        duration = get_video_duration(video_path)
        _report("Done!", 1.0)

        return {
            "num_frames": len(self.frame_infos),
            "num_clips": len(self.clips),
            "duration_sec": duration,
            "cached": False,
        }

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 5) -> dict:
        """
        Answer a natural-language question about the ingested video.

        Args:
            question: User's question string.
            k:        Number of clips to retrieve.

        Returns:
            Dict with:
                - 'clips'      : list of clip dicts (sorted by score desc)
                - 'explanation': LLM / template explanation string
        """
        if not self._ingested:
            raise RuntimeError("Call ingest() before query().")

        # Embed query
        query_emb = embed_text(question)

        # Retrieve top-k clips
        results = self.retriever.search(query_emb, k=k)

        # Generate explanation
        explanation = explain(question, results)

        return {
            "clips": results,
            "explanation": explanation,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ingested

    def reset(self):
        """Clear current state (call before ingesting a new video)."""
        self.retriever = VideoRetriever(dim=EMBEDDING_DIM)
        self.frame_infos = []
        self.clips = []
        self.video_path = None
        self._ingested = False
