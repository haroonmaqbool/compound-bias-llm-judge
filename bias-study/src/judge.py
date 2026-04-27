# src/judge.py
# Unified judge wrapper for all 5 models — $0 cost.
# Returns 'A' or 'B' (winner label) for each pairwise comparison.

import os, time
from groq import Groq, RateLimitError
import google.generativeai as genai
import ollama
from dotenv import load_dotenv

load_dotenv()

# ── Groq key pool (shared with variant_generator) ────
_raw_keys    = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
_api_keys    = [k.strip() for k in _raw_keys.split(",") if k.strip()]
_key_index   = 0
_groq_clients = [Groq(api_key=k) for k in _api_keys]

def _groq_client() -> Groq:
    return _groq_clients[_key_index]

def _rotate_key():
    global _key_index
    _key_index = (_key_index + 1) % len(_groq_clients)
    print(f"  ↻ Groq key rotated → key[{_key_index+1}/{len(_groq_clients)}]")

# ── Gemini ────────────────────────────────────────────
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

# ── Shared prompt ─────────────────────────────────────
JUDGE_PROMPT = """You are evaluating two responses to a question.
Question: {question}
Response A: {response_a}
Response B: {response_b}
Which response is better? Reply with only 'A' or 'B'."""


# ── Individual judge functions ────────────────────────
def judge_groq(model_id: str, question: str, resp_a: str, resp_b: str) -> str:
    global _key_index
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    for _ in range(len(_groq_clients)):
        try:
            result = _groq_client().chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
            )
            time.sleep(0.4)
            return result.choices[0].message.content.strip().upper()
        except RateLimitError:
            _rotate_key()
    return "ERROR"


def judge_gemini(question: str, resp_a: str, resp_b: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    time.sleep(4)  # 15 req/min hard limit → 4s gap keeps you safe
    response = gemini_model.generate_content(prompt)
    return response.text.strip().upper()


def judge_ollama(question: str, resp_a: str, resp_b: str) -> str:
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a, response_b=resp_b)
    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip().upper()


# ── Master dict — import this everywhere ─────────────
_M70B     = os.getenv("GROQ_JUDGE_70B",     "llama-3.3-70b-versatile")
_MMIXTRAL = os.getenv("GROQ_JUDGE_MIXTRAL", "mixtral-8x7b-32768")
_MGEMMA   = os.getenv("GROQ_JUDGE_GEMMA",   "gemma2-9b-it")

JUDGES = {
    "llama3-70b": lambda q, a, b: judge_groq(_M70B,     q, a, b),
    "mixtral":    lambda q, a, b: judge_groq(_MMIXTRAL, q, a, b),
    "gemma2":     lambda q, a, b: judge_groq(_MGEMMA,   q, a, b),
    "gemini":     judge_gemini,
    "llama3-8b":  judge_ollama,
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
    # Quick smoke test — replace with real question/responses
    q  = "What is the capital of France?"
    a  = "The capital of France is Paris."
    b  = "France's capital city is Paris, a major European metropolis."
    for name in JUDGES:
        verdict = call_judge(name, q, a, b)
        print(f"  {name:15s} → {verdict}")