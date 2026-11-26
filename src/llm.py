"""
Model hook-up for the whole project.

    from src.llm import get_llm
    llm = get_llm()                 # backend/model come from env vars (below)
    text = llm(prompt)              # prompt: str -> response: str
    texts = llm.batch([p1, p2])     # same, faster on the HF backend

Every part of the pipeline (evolution, baseline, retrieval check, applicability
check) only sees `llm(prompt) -> str`, so swapping models never touches them.
All decoding is greedy / temperature 0 so that runs are repeatable.

Backends
--------
gemini : Google Gemini API (needs GEMINI_API_KEY in .env). Rate-limited.
hf     : open-source model via Hugging Face Transformers on a GPU (use on ASU SOL).

Environment variables (all optional)
------------------------------------
LLM_BACKEND      "gemini" (default) or "hf"
HF_MODEL         default "Qwen/Qwen2.5-7B-Instruct"
GEMINI_MODEL     default "gemini-3.5-flash-lite"
MAX_NEW_TOKENS   default 1536  (cap per response; too small => truncated
                 reasoning => parse failures, so watch the parse-failure rate)
HF_BATCH_SIZE    default 8     (lower it if you hit CUDA out-of-memory)
HF_ALLOW_CPU     set to 1 to allow the hf backend without a GPU (very slow;
                 only for debugging)
HF_HOME          where Hugging Face caches weights. On SOL point this at scratch
                 so the weights are not stored in your home directory.

torch / transformers / google-genai are imported lazily, so importing this
module never fails just because one backend's package is missing.
"""

import os
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv is only needed for the Gemini key file
    pass

DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "gemini")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1536"))
HF_BATCH_SIZE = int(os.environ.get("HF_BATCH_SIZE", "8"))


def get_llm(model_name=None, backend=None):
    """
    Return `llm`, a function llm(prompt: str) -> str with an extra
    `llm.batch(prompts: list[str]) -> list[str]` attribute and a
    `llm.description` string (backend + model, for logging).
    """
    backend = (backend or DEFAULT_BACKEND).lower()

    if backend == "gemini":
        return _gemini_llm(model_name or DEFAULT_GEMINI_MODEL)
    if backend == "hf":
        return _hf_llm(model_name or DEFAULT_HF_MODEL)

    raise ValueError(f"Unknown backend '{backend}'. Use 'gemini' or 'hf'.")


# --------------------------------------------------------------------- Gemini


def _gemini_llm(model):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Add it to the project-root .env file."
        )

    client = genai.Client(api_key=api_key)

    # The free tier allows 15 requests per minute: one request every 4 seconds.
    # 4.5 s leaves a small safety margin. The lock keeps one request in flight.
    min_interval = 4.5
    lock = threading.Lock()
    last_request_time = [0.0]

    def llm(prompt):
        for attempt in range(4):
            with lock:
                wait = min_interval - (time.time() - last_request_time[0])
                if wait > 0:
                    time.sleep(wait)
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0),
                    )
                    last_request_time[0] = time.time()
                    return (response.text or "").strip()
                except Exception:  # rate limit, network blip, 5xx
                    last_request_time[0] = time.time()
                    if attempt == 3:
                        raise
            time.sleep(10 * (attempt + 1))

    llm.batch = lambda prompts: [llm(p) for p in prompts]
    llm.description = f"gemini:{model}"
    return llm


# ------------------------------------------------------------ Hugging Face / GPU


def _hf_llm(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and os.environ.get("HF_ALLOW_CPU") != "1":
        raise RuntimeError(
            "torch.cuda.is_available() is False, so no GPU is visible.\n"
            "On SOL: are you on a compute node (not the login node), did the job "
            "request a GPU, and is torch a CUDA build? Run "
            "`python scripts/check_gpu.py` to see which one is wrong.\n"
            "(Set HF_ALLOW_CPU=1 to run on CPU anyway; it is very slow.)"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # needed for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto"}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:  # newer transformers renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, **load_kwargs
        )
    model.eval()

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    lock = threading.Lock()

    def render(prompt):
        if not has_chat_template:
            return prompt
        # enable_thinking=False is ignored by templates that do not use it, and
        # stops Qwen3-style models from spending the token budget on <think>.
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(prompts):
        """One forward batch. Splits in half on CUDA out-of-memory."""
        texts = [render(p) for p in prompts]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not has_chat_template,  # template adds its own BOS
        ).to(model.device)

        try:
            with lock, torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # greedy
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            mid = len(prompts) // 2
            return generate(prompts[:mid]) + generate(prompts[mid:])

        new_tokens = output[:, encoded["input_ids"].shape[1]:]
        return [
            tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in new_tokens
        ]

    def batch(prompts):
        """Responses for many prompts, in the same order as `prompts`."""
        prompts = list(prompts)
        # Group similar lengths so padding is small, then restore the order.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        results = [None] * len(prompts)

        for start in range(0, len(order), HF_BATCH_SIZE):
            chunk = order[start:start + HF_BATCH_SIZE]
            for index, text in zip(chunk, generate([prompts[i] for i in chunk])):
                results[index] = text

        return results

    def llm(prompt):
        return generate([prompt])[0]

    llm.batch = batch
    llm.description = f"hf:{model_name}"
    return llm
