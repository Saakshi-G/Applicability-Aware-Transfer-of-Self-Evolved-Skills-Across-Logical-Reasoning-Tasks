import os
from pathlib import Path

import time
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loading API credentials from the project-root .env file.
load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

# Since the Gemini API is rate-limited, we use a lock to ensure that only one
# request is in-flight at a time. 

def get_llm(model_name=None):
    """
    Return a callable:

        llm(prompt: str) -> str

    Gemini free tier currently limits requests per minute.
    This wrapper spaces requests so the rest of the project
    does not need to know about API rate limits.
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Add it to the project-root .env file."
        )

    model = model_name or DEFAULT_MODEL

    client = genai.Client(api_key=api_key)

    lock = threading.Lock()
    last_request_time = [0.0]

    # 15 RPM = one request every 4 seconds.
    # Use 4.5 seconds for a small safety margin.
    min_interval = 4.5

    def llm(prompt):
        with lock:
            elapsed = time.time() - last_request_time[0]

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                ),
            )

            last_request_time[0] = time.time()

        if not response.text:
            return ""

        return response.text.strip()

    return llm

# def get_llm(model_name=None):
    """
    Return a callable with the interface:

        llm(prompt: str) -> str

    This is the interface expected by the skill-evolution pipeline.
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Add it to the project-root .env file."
        )

    model = model_name or DEFAULT_MODEL

    client = genai.Client(api_key=api_key)

    def llm(prompt):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
            ),
        )

        if not response.text:
            return ""

        return response.text.strip()

    return llm