"""
llm.py
LLM explanation layer for Video QA results.
Tries Gemini first, then OpenAI, then falls back to a structured template answer.
"""

import logging
import os

from dotenv import load_dotenv
from PIL import Image

from video_qa.temporal import format_timestamp

logger = logging.getLogger(__name__)

# Load .env automatically (if present) so API keys work with `streamlit run app.py`
load_dotenv(override=False)


def _build_prompt(query: str, clips: list[dict]) -> str:
    """Build a prompt string from retrieved clips."""
    clip_descriptions = []
    for i, clip in enumerate(clips, 1):
        ts_start = format_timestamp(clip["start_sec"])
        ts_end = format_timestamp(clip["end_sec"])
        score = clip.get("score", 0.0)
        clip_descriptions.append(
            f"  Clip {i}: [{ts_start} → {ts_end}]  (relevance score: {score:.3f})"
        )

    clips_text = "\n".join(clip_descriptions)

    return f"""You are an intelligent video analysis assistant.
A user asked the following question about a video:

Question: "{query}"

The retrieval system identified these most relevant video segments (sorted by relevance):

{clips_text}

Based on these timestamps and their relevance scores, provide a concise, helpful answer.
- Mention the specific timestamps where the event likely occurs.
- If multiple clips are relevant, explain the temporal progression.
- Keep your answer to 3–5 sentences.
"""


def _build_gemini_parts(query: str, clips: list[dict]) -> list:
    """Build multimodal Gemini input: text + representative frames for top clips."""
    top = clips[:5]

    lines = [
        "You are an intelligent video analysis assistant.",
        f'Question: "{query}"',
        "You are given timestamps and representative frames from the top retrieved clips.",
        "Use this evidence to answer in 3-5 sentences.",
        "Mention timestamps in your final answer.",
        "If uncertain, state uncertainty briefly.",
        "",
        "Retrieved clips:",
    ]

    for i, clip in enumerate(top, 1):
        ts_start = format_timestamp(clip["start_sec"])
        ts_end = format_timestamp(clip["end_sec"])
        score = clip.get("score", 0.0)
        lines.append(f"- Clip {i}: [{ts_start} → {ts_end}] (score: {score:.3f})")

    parts: list = ["\n".join(lines)]

    for i, clip in enumerate(top, 1):
        frame_path = clip.get("representative_frame", "")
        if not frame_path or not os.path.exists(frame_path):
            continue
        try:
            img = Image.open(frame_path).convert("RGB")
            ts_start = format_timestamp(clip["start_sec"])
            ts_end = format_timestamp(clip["end_sec"])
            parts.append(f"Representative frame for Clip {i} at {ts_start} → {ts_end}:")
            parts.append(img)
        except Exception as e:
            logger.warning(f"Failed to load representative frame '{frame_path}': {e}")

    return parts


def _try_gemini(prompt: str, query: str, clips: list[dict]) -> str | None:
    """Attempt to call Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.info("GEMINI_API_KEY not found in environment; skipping Gemini.")
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        parts = _build_gemini_parts(query, clips)
        response = model.generate_content(parts if parts else prompt)

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        # Fallback parsing for SDK variants
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            chunks = getattr(content, "parts", None) or []
            joined = "\n".join(
                p.text for p in chunks if getattr(p, "text", None)
            ).strip()
            if joined:
                return joined

        logger.warning("Gemini returned an empty response.")
        return None
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return None


def _try_openai(prompt: str) -> str | None:
    """Attempt to call OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return None


def _template_answer(query: str, clips: list[dict]) -> str:
    """Fallback: structured template answer with timestamps."""
    if not clips:
        return "No relevant segments found for your query."

    top = clips[0]
    ts_start = format_timestamp(top["start_sec"])
    ts_end = format_timestamp(top["end_sec"])

    lines = [
        f"Based on visual retrieval, the most relevant segment for \"{query}\" "
        f"appears at **{ts_start} → {ts_end}** (similarity: {top.get('score', 0):.3f}).",
    ]

    if len(clips) > 1:
        other_times = ", ".join(
            f"{format_timestamp(c['start_sec'])}→{format_timestamp(c['end_sec'])}"
            for c in clips[1:]
        )
        lines.append(f"Other potentially relevant moments: {other_times}.")

    lines.append(
        "💡 Add a GEMINI_API_KEY or OPENAI_API_KEY environment variable for AI-generated explanations."
    )

    return " ".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def explain(query: str, clips: list[dict]) -> str:
    """
    Generate a natural-language explanation for the retrieved clips.

    Tries: Gemini → OpenAI → Template fallback.

    Args:
        query: The user's question.
        clips: Retrieved clip dicts (with start_sec, end_sec, score).

    Returns:
        Explanation string.
    """
    prompt = _build_prompt(query, clips)

    answer = _try_gemini(prompt, query, clips)
    if answer:
        logger.info("Used Gemini for explanation.")
        return answer

    answer = _try_openai(prompt)
    if answer:
        logger.info("Used OpenAI for explanation.")
        return answer

    logger.info("Falling back to template explanation.")
    return _template_answer(query, clips)
