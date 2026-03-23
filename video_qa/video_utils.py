"""
video_utils.py
Frame extraction with precise timestamp mapping.
"""

import cv2
import os
import logging

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, output_folder: str = "frames", fps: float = 1.0) -> list[dict]:
    """
    Extract frames from a video at a given sample rate.

    Args:
        video_path:     Path to input video file.
        output_folder:  Directory to save extracted frames.
        fps:            How many frames to sample per second (default 1).

    Returns:
        List of dicts with keys:
            - frame_path   (str)   : absolute path to saved JPEG
            - frame_number (int)   : original frame index in video
            - timestamp_sec (float): time in seconds of that frame
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0  # fallback

    interval = max(1, int(round(video_fps / fps)))

    frame_infos = []
    frame_number = 0
    saved = 0

    logger.info(f"Video FPS: {video_fps:.2f} | Sample every {interval} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % interval == 0:
            timestamp_sec = frame_number / video_fps
            filename = os.path.join(output_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(filename, frame)
            frame_infos.append({
                "frame_path": os.path.abspath(filename),
                "frame_number": frame_number,
                "timestamp_sec": round(timestamp_sec, 3),
                "frame_index": saved,   # sequential saved index
            })
            saved += 1

        frame_number += 1

    cap.release()
    logger.info(f"Extracted {saved} frames from '{video_path}'")
    return frame_infos


def get_video_duration(video_path: str) -> float:
    """Return total duration of video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps
