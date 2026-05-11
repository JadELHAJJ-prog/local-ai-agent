import subprocess
import tempfile
import os
import base64

import cv2
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from config import SANDBOX_IMAGE, MAX_FRAMES
from models import vlm
from ddgs import DDGS


@tool
def search_web(query: str) -> str:
    """Search the web for current information, recent events,
    or anything that requires up-to-date knowledge beyond
    your training data. Input should be a search query string."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n".join(
            [
                f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}"
                for r in results
            ]
        )


@tool
def execute_code(code: str) -> str:
    """Execute Python code safely in an isolated Docker container.
    Use this when the user asks to run code, perform calculations,
    or test a Python script. Input should be valid Python code."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "128m",
                "--cpus", "0.5",
                "-v", f"{tmp_path}:/app/code.py:ro",
                SANDBOX_IMAGE, "python", "/app/code.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return result.stdout or "Code executed successfully with no output."
        else:
            return f"Error:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@tool
def analyze_image(image_path: str, question: str = "What is in this image?") -> str:
    """Analyze an image using vision AI. Use this when the user
    provides an image path and wants to know what's in it,
    extract text from it, or ask questions about it.
    Input should be the path to the image file."""
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )

    response = vlm.invoke([message])
    return response.content if response.content else "No response from vision model."


@tool
def analyze_video(
    video_path: str,
    question: str = "What is happening in this video?",
) -> str:
    """Analyze a video using vision AI by sampling key frames.
    Use this when the user provides a video file path (.mp4, .avi, .mov, .mkv).
    Input should be the path to the video file."""
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Could not open video file at {video_path}"

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // MAX_FRAMES)
        frames_analyzed = 0
        content = [{"type": "text", "text": question}]

        for frame_idx in range(0, total_frames, interval):
            if frames_analyzed >= MAX_FRAMES:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            )
            frames_analyzed += 1
            print(f"Extracted frame {frames_analyzed}/{MAX_FRAMES}...", flush=True)

        if frames_analyzed == 0:
            return "No frames could be extracted from this video."

        print("Sending frames to vision model...", flush=True)
        response = vlm.invoke([HumanMessage(content=content)])
        return response.content or "No response from vision model."

    finally:
        cap.release()


tools = [search_web, execute_code, analyze_image, analyze_video]
