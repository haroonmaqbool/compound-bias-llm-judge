# src/judge.py
# Unified judge wrapper for all 5 models.
# Returns 'A' or 'B' (winner label) for each pairwise comparison.
#
# Judge roster (all free-tier / low-cost):
#   1. llama3-70b   — Groq (llama-3.3-70b-versatile)
#   2. qwen3-32b    — Groq (qwen/qwen3-32b)
#   3. llama4-scout — Groq (meta-llama/llama-4-scout-17b-16e-instruct)
#   4. gemini-flash — Google Gemini 2.0 Flash (google-genai SDK)
#   5. llama3-8b    — Groq (llama-3.1-8b-instant)

import os, re, time
from groq import Groq, RateLimitError
from google import genai as google_genai
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

# ── Gemini (google-genai SDK) ─────────────────────────────────────────────────
_gemini_client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

# ── Shared judge prompt ───────────────────────────────────────────────────────
JUDGE_PROMPT = """You are evaluating two responses to a question.
Question: {question}
Response A: {response_a}
Response B: {response_b}
Which response is better? Reply with only 'A' or 'B'."""


# ── Individual judge functions ────────────────────────────────────────────────
def judge_groq(model_id: str, question: str, resp_a: str, resp_b: str) -> str:
    global _key_index
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
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
            # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Extract first A or B token
            m = re.search(r"\b([AB])\b", raw.upper())
            return m.group(1) if m else raw.upper()
        except RateLimitError:
            _rotate_key()
    return "ERROR"


def judge_gemini(question: str, resp_a: str, resp_b: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    try:
        time.sleep(4)   # 15 req/min hard limit → 4s gap keeps us safe
        response = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        raw = response.text.strip().upper()
        m = re.search(r"\b([AB])\b", raw)
        return m.group(1) if m else raw
    except Exception as e:
        err = str(e)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            # Daily quota hit — fall back to Groq gemma3
            return judge_groq("gemma-3-27b-it" if False else _MGEMMA_FB, question, resp_a, resp_b)
        raise


# ── Model IDs from .env ───────────────────────────────────────────────────────
_M70B     = os.getenv("GROQ_JUDGE_70B",      "llama-3.3-70b-versatile")
_MQWEN    = os.getenv("GROQ_JUDGE_QWEN",     "qwen/qwen3-32b")
_MLLAMA4  = os.getenv("GROQ_JUDGE_LLAMA4",   "meta-llama/llama-4-scout-17b-16e-instruct")
_M8B      = os.getenv("GROQ_JUDGE_8B",       "llama-3.1-8b-instant")
_MGEMMA_FB = os.getenv("GROQ_JUDGE_GEMMA_FB", "llama-3.1-8b-instant")  # Gemini quota fallback

# ── Master judge dict — import this everywhere ────────────────────────────────
JUDGES = {
    "llama3-70b":   lambda q, a, b: judge_groq(_M70B,    q, a, b),
    "qwen3-32b":    lambda q, a, b: judge_groq(_MQWEN,   q, a, b),
    "llama4-scout": lambda q, a, b: judge_groq(_MLLAMA4, q, a, b),
    "gemini-flash": judge_gemini,
    "llama3-8b":    lambda q, a, b: judge_groq(_M8B,     q, a, b),
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
    b = "France's capital city is Paris, a major European metropolis."
    for name in JUDGES:
        verdict = call_judge(name, q, a, b)
        print(f"  {name:15s} -> {verdict}")
