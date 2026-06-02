# src/judge.py  — week3-fixes
# Unified judge wrapper for all 5 models.
# Returns 'A' or 'B' (winner label) for each pairwise comparison.
#
# Changes vs week2:
#   • Gemini (quota-limited, silent fallback bug) → replaced with Mistral Small 3.1
#   • llama3-70b (Groq rate-limit) → replaced with DeepSeek V4-Flash
#   • Gemini fallback bug fixed: was silently calling llama3-8b under gemini-flash label;
#     new implementation raises ERROR cleanly if any judge quota-fails.
#   • Both new judges use OpenAI-compatible SDK (single dependency: openai package)
#
# Final judge roster (all free-tier or pennies-per-run):
#   1. deepseek-flash — DeepSeek V4-Flash  ($0.14/1M in,  replaces llama3-70b)
#   2. qwen3-32b      — Groq (qwen/qwen3-32b)               [unchanged]
#   3. llama4-scout   — Groq (meta-llama/llama-4-scout-17b) [unchanged]
#   4. mistral-small  — Mistral Small 3.1  ($0.10/1M in,  replaces gemini-flash)
#   5. llama3-8b      — Groq (llama-3.1-8b-instant)         [unchanged]

import os, re, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from groq import Groq, RateLimitError
from openai import OpenAI         # used for DeepSeek + Mistral (both OAI-compatible)
from dotenv import load_dotenv

load_dotenv()

# ── Groq key pool ─────────────────────────────────────────────────────────────
_raw_keys     = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
_api_keys     = [k.strip() for k in _raw_keys.split(",") if k.strip()]
_key_index    = 0
_groq_clients = [Groq(api_key=k) for k in _api_keys]

def _groq_client() -> Groq:
    return _groq_clients[_key_index]

def _rotate_key():
    global _key_index
    _key_index = (_key_index + 1) % len(_groq_clients)
    print(f"  ↻ Groq key rotated → key[{_key_index+1}/{len(_groq_clients)}]")

# ── DeepSeek client (OpenAI-compatible) ───────────────────────────────────────
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")   # V4-Flash maps to "deepseek-chat"
_deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
)

# ── Mistral client (OpenAI-compatible) ────────────────────────────────────────
_MISTRAL_MODEL  = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
_mistral_client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
    base_url="https://api.mistral.ai/v1",
)

# ── Shared judge prompt ───────────────────────────────────────────────────────
JUDGE_PROMPT = """You are evaluating two responses to a question.
Question: {question}
Response A: {response_a}
Response B: {response_b}
Which response is better? Reply with only 'A' or 'B'."""


# ── Helper: extract A or B from raw model output ─────────────────────────────
def _extract_verdict(raw: str) -> str:
    """Strip thinking tags and pull first A or B token."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\b([AB])\b", raw.upper())
    return m.group(1) if m else raw.upper()


# ── Individual judge functions ────────────────────────────────────────────────
def judge_groq(model_id: str, question: str, resp_a: str, resp_b: str) -> str:
    global _key_index
    prompt  = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    # qwen3 needs more tokens to finish its <think> block before emitting A/B
    max_tok = 1024 if "qwen" in model_id.lower() else 10
    for _ in range(len(_groq_clients)):
        try:
            result = _groq_client().chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
            )
            time.sleep(0.4)
            raw = result.choices[0].message.content
            return _extract_verdict(raw)
        except RateLimitError:
            _rotate_key()
    return "ERROR"


def judge_deepseek(question: str, resp_a: str, resp_b: str) -> str:
    """DeepSeek V4-Flash via OpenAI-compatible API.  ~$0.14 / 1M input tokens."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    try:
        result = _deepseek_client.chat.completions.create(
            model=_DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        time.sleep(0.2)
        raw = result.choices[0].message.content
        return _extract_verdict(raw)
    except Exception as e:
        print(f"  [WARN] DeepSeek failed: {e}")
        return "ERROR"


def judge_mistral(question: str, resp_a: str, resp_b: str) -> str:
    """Mistral Small 3.1 via OpenAI-compatible API.  ~$0.10 / 1M input tokens."""
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    try:
        result = _mistral_client.chat.completions.create(
            model=_MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        time.sleep(0.2)
        raw = result.choices[0].message.content
        return _extract_verdict(raw)
    except Exception as e:
        print(f"  [WARN] Mistral failed: {e}")
        return "ERROR"


# ── Model IDs from .env ───────────────────────────────────────────────────────
_MQWEN   = os.getenv("GROQ_JUDGE_QWEN",   "qwen/qwen3-32b")
_MLLAMA4 = os.getenv("GROQ_JUDGE_LLAMA4", "meta-llama/llama-4-scout-17b-16e-instruct")
_M8B     = os.getenv("GROQ_JUDGE_8B",     "llama-3.1-8b-instant")

# ── Master judge dict — import this everywhere ────────────────────────────────
JUDGES = {
    "deepseek-flash":  judge_deepseek,                              # replaces llama3-70b
    "qwen3-32b":       lambda q, a, b: judge_groq(_MQWEN,   q, a, b),
    "llama4-scout":    lambda q, a, b: judge_groq(_MLLAMA4, q, a, b),
    "mistral-small":   judge_mistral,                               # replaces gemini-flash
    "llama3-8b":       lambda q, a, b: judge_groq(_M8B,     q, a, b),
}


def call_judge(judge_name: str, question: str, resp_a: str, resp_b: str) -> str:
    """Safe wrapper — returns 'A', 'B', or 'ERROR'."""
    try:
        result = JUDGES[judge_name](question, resp_a, resp_b)
        return result if result in ("A", "B") else "INVALID"
    except Exception as e:
        print(f"    [WARN] {judge_name} failed: {e}")
        return "ERROR"


if __name__ == "__main__":
    q = "What is the capital of France?"
    a = "The capital of France is Paris."
    b = "France's capital city is Paris, a major European metropolis known for the Eiffel Tower."
    print("Smoke-testing all 5 judges...\n")
    for name in JUDGES:
        verdict = call_judge(name, q, a, b)
        print(f"  {name:18s} → {verdict}")
