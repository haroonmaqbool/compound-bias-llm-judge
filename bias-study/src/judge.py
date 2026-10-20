# src/judge.py — week4-hybrid
# ================================================================
# Unified judge wrapper for the 5-model hybrid roster:
#
#   LOCAL  (Ollama, 16 GB VRAM, one model at a time):
#     1. qwen2.5    — qwen2.5:14b     (heavyweight, best quality)
#     2. llama3-8b  — llama3:8b       (lightweight, fast)
#     3. gemma2-9b  — gemma2:9b       (mid-weight)
#     4. mistral    — mistral:latest  (7B, fully local)
#
#   CLOUD  (Groq API — auto-rotates across key pool on 429):
#     5. qwen3-32b  — qwen/qwen3-32b
#
# Design decisions:
#   • Ollama models use keep_alive=0 so VRAM is freed after each call.
#   • Groq judges use the existing key-rotation pool (6 keys in .env).
#   • qwen3-32b needs up to 1024 tokens to finish its <think> block.
#   • check_all_judges() probes every endpoint and reports pass/fail.
# ================================================================

import os, re, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Ollama config ─────────────────────────────────────────────────────────────
_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OL_QWEN     = os.getenv("DEEPSEEK_MODEL",     "qwen2.5:14b")
_OL_LLAMA    = os.getenv("MISTRAL_MODEL",       "llama3:8b")
_OL_GEMMA    = os.getenv("GROQ_VARIANT_MODEL",  "gemma2:9b")

# ── Groq key pool ─────────────────────────────────────────────────────────────
_raw_keys     = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
_api_keys     = [k.strip() for k in _raw_keys.split(",") if k.strip()]
_key_index    = 0
_groq_clients = [Groq(api_key=k) for k in _api_keys]

_GROQ_QWEN  = os.getenv("GROQ_JUDGE_QWEN", "qwen/qwen3-32b")
_OL_MISTRAL  = os.getenv("GROQ_JUDGE_8B",   "mistral:latest")

if not _api_keys:
    print("[WARN] No Groq API keys found in .env — Groq judges will return ERROR.")


def _groq_client() -> Groq:
    return _groq_clients[_key_index]


def _rotate_key():
    global _key_index
    _key_index = (_key_index + 1) % len(_groq_clients)
    print(f"  ↻ Groq key rotated → key[{_key_index+1}/{len(_groq_clients)}]")


# ── Shared judge prompt ───────────────────────────────────────────────────────
JUDGE_PROMPT = """You are evaluating two responses to a question.
Question: {question}
Response A: {response_a}
Response B: {response_b}
Which response is better? Reply with only 'A' or 'B'."""


# ── Helper: extract A or B from raw model output ──────────────────────────────
def _extract_verdict(raw: str) -> str:
    """Strip thinking/reasoning tags and pull first A or B token."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"<\|.*?\|>", "", raw).strip()
    m = re.search(r"\b([AB])\b", raw.upper())
    return m.group(1) if m else "INVALID"


# ── OLLAMA: shared call helper ────────────────────────────────────────────────
def _ollama_chat(model_tag: str, prompt: str, timeout: int = 120) -> str:
    """
    POST to Ollama /api/chat.
    keep_alive=0  →  model is unloaded from VRAM immediately after the call.
    This keeps memory free for the next model on a 16 GB card.
    """
    url     = f"{_OLLAMA_BASE}/api/chat"
    payload = {
        "model":    model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  {"temperature": 0, "num_predict": 16},
        "keep_alive": 60,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _extract_verdict(content)
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Ollama not running or model '{model_tag}' unavailable. "
              f"Run: ollama serve && ollama pull {model_tag}")
        return "ERROR"
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Ollama timed out for '{model_tag}' (>{timeout}s).")
        return "ERROR"
    except Exception as e:
        print(f"  [ERROR] Ollama '{model_tag}': {e}")
        return "ERROR"


# ── GROQ: shared call helper ──────────────────────────────────────────────────
def _groq_chat(model_id: str, prompt: str) -> str:
    """Call a Groq model with automatic key rotation on rate-limit errors."""
    global _key_index
    if not _groq_clients:
        print("  [ERROR] No Groq API keys configured.")
        return "ERROR"

    # qwen3 needs space to finish its <think> block before emitting A/B
    max_tok = 1024 if "qwen" in model_id.lower() else 16

    for _ in range(len(_groq_clients)):
        try:
            result = _groq_client().chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=0,
            )
            time.sleep(0.4)
            raw = result.choices[0].message.content or ""
            return _extract_verdict(raw)
        except RateLimitError:
            _rotate_key()
        except Exception as e:
            print(f"  [WARN] Groq '{model_id}': {e}")
            return "ERROR"
    print(f"  [ERROR] All Groq keys exhausted for '{model_id}'.")
    return "ERROR"


# ── Individual judge functions ────────────────────────────────────────────────

def judge_qwen25(question: str, resp_a: str, resp_b: str) -> str:
    """qwen2.5:14b via local Ollama."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    return _ollama_chat(_OL_QWEN, prompt)


def judge_llama3_8b(question: str, resp_a: str, resp_b: str) -> str:
    """llama3:8b via local Ollama."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    return _ollama_chat(_OL_LLAMA, prompt)


def judge_gemma2_9b(question: str, resp_a: str, resp_b: str) -> str:
    """gemma2:9b via local Ollama."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    return _ollama_chat(_OL_GEMMA, prompt)


def judge_qwen3_32b(question: str, resp_a: str, resp_b: str) -> str:
    """qwen/qwen3-32b via Groq API."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    return _groq_chat(_GROQ_QWEN, prompt)


def judge_mistral(question: str, resp_a: str, resp_b: str) -> str:
    """mistral:latest via local Ollama."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    return _ollama_chat(_OL_MISTRAL, prompt)


# ── Master judge dict — import this everywhere ────────────────────────────────
JUDGES = {
    "qwen2.5":   judge_qwen25,       # Ollama local  — 14B heavyweight
    "llama3-8b": judge_llama3_8b,    # Ollama local  — 8B lightweight
    "gemma2-9b": judge_gemma2_9b,    # Ollama local  — 9B mid-weight
    "mistral":   judge_mistral,      # Ollama local  — 7B Mistral
    "qwen3-32b": judge_qwen3_32b,    # Groq cloud    — 32B
}


def call_judge(judge_name: str, question: str, resp_a: str, resp_b: str) -> str:
    """Safe wrapper — always returns 'A', 'B', 'INVALID', or 'ERROR'."""
    if judge_name not in JUDGES:
        print(f"  [ERROR] Unknown judge '{judge_name}'. Valid: {list(JUDGES.keys())}")
        return "ERROR"
    try:
        result = JUDGES[judge_name](question, resp_a, resp_b)
        return result if result in ("A", "B") else "INVALID"
    except Exception as e:
        print(f"  [ERROR] {judge_name}: {e}")
        return "ERROR"


# ── API key + endpoint checker ────────────────────────────────────────────────

def check_all_judges(verbose: bool = True) -> dict:
    """
    Probe every judge with a trivial question.
    Returns {judge_name: True/False}.
    Prints a clear PASS/FAIL report so you know which keys to remove from .env.
    """
    q  = "What is 2 + 2?"
    a1 = "4"
    a2 = "The answer is four (4)."

    results = {}
    print("\n" + "=" * 62)
    print("  Judge endpoint check")
    print("=" * 62)
    print(f"  Groq keys loaded: {len(_groq_clients)}")
    print(f"  Ollama base:      {_OLLAMA_BASE}")
    print("-" * 62)

    BACKEND = {
        "qwen2.5":   "OLLAMA",
        "llama3-8b": "OLLAMA",
        "gemma2-9b": "OLLAMA",
        "mistral":   "OLLAMA",
        "qwen3-32b": "GROQ  ",
    }

    for name, fn in JUDGES.items():
        try:
            verdict = fn(q, a1, a2)
            ok      = verdict in ("A", "B")
            status  = "✓ PASS" if ok else f"✗ FAIL (got: {verdict!r})"
        except Exception as e:
            ok     = False
            status = f"✗ ERROR: {e}"
        results[name] = ok
        print(f"  [{BACKEND[name]}] {name:14s}  →  {status}")

    print("=" * 62)
    passing = [k for k, v in results.items() if v]
    failing = [k for k, v in results.items() if not v]
    print(f"  Passing ({len(passing)}): {passing}")
    if failing:
        print(f"  ⚠ Failing ({len(failing)}): {failing}")
        print()
        for f in failing:
            if BACKEND[f].strip() == "OLLAMA":
                model = {"qwen2.5": _OL_QWEN, "llama3-8b": _OL_LLAMA,
                         "gemma2-9b": _OL_GEMMA, "mistral": _OL_MISTRAL}[f]
                print(f"    Fix '{f}': ollama pull {model}")
            else:
                print(f"    Fix '{f}': check GROQ_API_KEYS in .env")

    print("=" * 62 + "\n")
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_all_judges()
