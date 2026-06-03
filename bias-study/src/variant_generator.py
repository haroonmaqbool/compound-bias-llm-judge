# src/variant_generator.py — week4-hybrid
# Generates short/long variants of MT-Bench responses via LOCAL Ollama.
# No API keys needed — everything runs on your 16 GB GPU.
#
# Model: qwen2.5:14b  (set DEEPSEEK_MODEL in .env to change)
# keep_alive=0 on each call so VRAM is freed before the next model loads.
#
# ✓ Resume-safe   — skips pairs already written to output_path
# ✓ Batch logging — saves every BATCH_SIZE pairs and prints progress
# ✓ try/except    — logs failures per-variant, doesn't crash the whole run

import os, json, time, re
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL       = os.getenv("DEEPSEEK_MODEL",  "qwen2.5:14b")   # heavyweight local
BATCH_SIZE  = 5       # print progress every N pairs
TIMEOUT     = 180     # seconds — large model on long text may be slow


def _check_ollama():
    """Verify Ollama is reachable and the variant model is available."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in tags:
            print(f"\n  [WARN] Model '{MODEL}' not found in Ollama.")
            print(f"  Available models: {tags}")
            print(f"  Run:  ollama pull {MODEL}\n")
            return False
        print(f"  ✓ Ollama reachable — model '{MODEL}' ready.")
        return True
    except requests.exceptions.ConnectionError:
        print(f"\n  [ERROR] Cannot connect to Ollama at {OLLAMA_BASE}.")
        print("  Make sure Ollama is running:  ollama serve\n")
        return False


def generate_variant(text: str, style: str) -> str:
    """
    Generate a short (~50%) or long (~150%) rewrite of `text` via Ollama.
    Returns the rewritten text, or an empty string on failure.
    """
    instruction = (
        "Rewrite the following text at roughly half the length. "
        "Preserve all key facts and meaning. Return ONLY the rewritten text."
        if style == "short" else
        "Rewrite the following text at roughly 1.5× the length. "
        "Add relevant detail and explanation without inventing new facts. "
        "Return ONLY the rewritten text."
    )

    url     = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user",   "content": text},
        ],
        "stream":     False,
        "options":    {"temperature": 0.3, "num_predict": 1024},
        "keep_alive": 0,          # unload after call — don't hog VRAM
    }

    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        if not content:
            print(f"  [WARN] Empty response from Ollama for style='{style}'")
        return content
    except requests.exceptions.Timeout:
        print(f"  [WARN] Ollama timed out generating '{style}' variant (>{TIMEOUT}s). "
              f"Returning original text.")
        return text   # fallback: use original so the pair isn't lost
    except Exception as e:
        print(f"  [WARN] Ollama error for style='{style}': {e}")
        return text


# ── Resume helper ─────────────────────────────────────────────────────────────

def _count_existing(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_pairs(
    input_path:  str = "data/mt_bench.json",
    output_path: str = "data/pairs.jsonl",
    limit:       int = 80,
):
    """
    Read raw MT-Bench records, extract question + response pair,
    generate short/long variants via local Ollama, and save to output_path.

    Resume-safe: pairs already in output_path are skipped.
    """
    # Verify Ollama is up before starting long runs
    if not _check_ollama():
        raise RuntimeError("Ollama is not available. Cannot generate variants.")

    already_done = _count_existing(output_path)
    if already_done:
        print(f"  ↻ Resuming — {already_done} pairs already saved, skipping them.")

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records   = records[:limit]
    new_pairs = 0

    for i, record in enumerate(records):
        if i < already_done:
            continue

        question = record["conversation_a"][0]["content"]

        resp_a_raw = next(
            (m["content"] for m in reversed(record["conversation_a"]) if m["role"] == "assistant"),
            ""
        )
        resp_b_raw = next(
            (m["content"] for m in reversed(record["conversation_b"]) if m["role"] == "assistant"),
            ""
        )

        if not resp_a_raw or not resp_b_raw:
            print(f"  Skipping record {i} — missing response")
            continue

        print(f"  [{i+1}/{len(records)}] Generating 4 variants via Ollama ({MODEL})...")

        pair = {
            "question": question,
            "short_a":  generate_variant(resp_a_raw, "short"),
            "long_a":   generate_variant(resp_a_raw, "long"),
            "short_b":  generate_variant(resp_b_raw, "short"),
            "long_b":   generate_variant(resp_b_raw, "long"),
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair) + "\n")

        new_pairs += 1

        if new_pairs % BATCH_SIZE == 0:
            total_so_far = already_done + new_pairs
            print(f"\n  ── Checkpoint: {total_so_far}/{limit} pairs saved ──\n")

    total = already_done + new_pairs
    print(f"\n✓ Done. {new_pairs} new pairs — {total}/{limit} total in {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_pairs(
        input_path  = "data/mt_bench.json",
        output_path = "data/pairs.jsonl",
        limit       = 80,
    )