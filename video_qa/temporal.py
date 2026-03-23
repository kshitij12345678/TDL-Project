"""
temporal.py
Clip-based temporal reasoning: groups frames into time windows
and produces averaged embeddings for temporal understanding.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def group_into_clips(
    frame_infos: list[dict],
    clip_duration_sec: float = 2.0,
) -> list[dict]:
    """
    Group frame metadata into fixed-duration temporal clips.

    Args:
        frame_infos:       List of frame dicts (from video_utils.extract_frames).
        clip_duration_sec: Duration of each clip window in seconds.

    Returns:
        List of clip dicts:
            - clip_id           (int)
            - start_sec         (float)
            - end_sec           (float)
            - frame_indices     (list[int])  — positions in frame_infos list
            - representative_frame (str)    — path to middle frame of clip
    """
    if not frame_infos:
        return []

    clips = []
    clip_id = 0

    # Build clips by scanning through frames
    i = 0
    while i < len(frame_infos):
        clip_start = frame_infos[i]["timestamp_sec"]
        clip_end = clip_start + clip_duration_sec

        indices_in_clip = []
        j = i
        while j < len(frame_infos) and frame_infos[j]["timestamp_sec"] < clip_end:
            indices_in_clip.append(j)
            j += 1

        if not indices_in_clip:
            i += 1
            continue

        # Representative frame = middle frame of clip
        mid = indices_in_clip[len(indices_in_clip) // 2]
        actual_end = frame_infos[indices_in_clip[-1]]["timestamp_sec"]

        clips.append({
            "clip_id": clip_id,
            "start_sec": round(clip_start, 3),
            "end_sec": round(actual_end, 3),
            "frame_indices": indices_in_clip,
            "representative_frame": frame_infos[mid]["frame_path"],
            "num_frames": len(indices_in_clip),
        })

        clip_id += 1
        i = j  # advance past this clip's frames

    logger.info(f"Grouped {len(frame_infos)} frames → {len(clips)} clips "
                f"(clip_duration={clip_duration_sec}s)")
    return clips


def compute_clip_embeddings(
    clips: list[dict],
    frame_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Average frame embeddings within each clip, then L2-normalize.

    Args:
        clips:            Output of group_into_clips().
        frame_embeddings: Shape (N, D) — per-frame CLIP embeddings (already normalized).

    Returns:
        np.ndarray of shape (num_clips, D), float32, L2-normalized.
    """
    clip_embeddings = []
    for clip in clips:
        idxs = clip["frame_indices"]
        avg = frame_embeddings[idxs].mean(axis=0)
        clip_embeddings.append(avg)

    result = np.stack(clip_embeddings, axis=0).astype(np.float32)

    # Re-normalize after averaging
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (result / norms).astype(np.float32)


def format_timestamp(seconds: float) -> str:
    """Format seconds to human-readable MM:SS.mmm string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"
