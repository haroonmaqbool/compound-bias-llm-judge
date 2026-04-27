# src/variant_generator.py
# Reads raw MT-Bench records and produces short/long variants via Groq.
# Output: data/pairs.jsonl  (one JSON object per line)
#
# ✓ Multi-key rotation — switches key immediately on 429, no wasted wait time
# ✓ Resume-safe        — skips pairs already written to output_path
# ✓ Batch processing   — saves every BATCH_SIZE pairs and prints progress
# ✓ All models via .env (GROQ_VARIANT_MODEL)

import os, json, time, re
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Key pool ──────────────────────────────────────────────────────────────────
# Read all keys from GROQ_API_KEYS (comma-separated).
# Falls back to legacy GROQ_API_KEY if the new var isn't set.
_raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
API_KEYS  = [k.strip() for k in _raw_keys.split(",") if k.strip()]

if not API_KEYS:
    raise EnvironmentError("No Groq API keys found. Set GROQ_API_KEYS in .env")

MODEL      = os.getenv("GROQ_VARIANT_MODEL", "llama-3.3-70b-versatile")
BATCH_SIZE = 10          # print a summary line every N pairs
MAX_RETRIES_PER_KEY = 2  # attempts on each key before rotating to the next


# ── Key rotation pool ─────────────────────────────────────────────────────────

class KeyPool:
    """Cycles through Groq API keys; rotates on 429 rate-limit errors."""

    def __init__(self, keys: list[str]):
        self.keys    = keys
        self.index   = 0
        self.clients = [Groq(api_key=k) for k in keys]
        print(f"  ✓ Loaded {len(keys)} Groq API key(s)")

    @property
    def current(self) -> Groq:
        return self.clients[self.index]

    @property
    def current_label(self) -> str:
        return f"key[{self.index+1}/{len(self.keys)}] ...{self.keys[self.index][-6:]}"

    def rotate(self) -> bool:
        """Move to the next key. Returns False if we've exhausted all keys."""
        next_idx = (self.index + 1) % len(self.keys)
        if next_idx == self.index:          # only 1 key — can't rotate
            return False
        if next_idx < self.index:           # wrapped around — all keys tried
            return False
        self.index = next_idx
        print(f"  ↻ Rotated to {self.current_label}")
        return True

    def reset(self):
        """Reset rotation index (call after a successful run or new day)."""
        self.index = 0


pool = KeyPool(API_KEYS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_wait_seconds(error_message: str) -> float:
    """Extract suggested wait time from a Groq 429 message (e.g. '6m38.304s')."""
    m = re.search(r"(\d+)m([\d.]+)s", error_message)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"([\d.]+)s", error_message)
    if m:
        return float(m.group(1))
    return 60.0   # safe default


def _count_existing(output_path: str) -> int:
    """Count valid lines already saved in output_path (for resume)."""
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ── Core variant generator ────────────────────────────────────────────────────

def generate_variant(text: str, style: str) -> str:
    """
    Generate a short (~50%) or long (~150%) rewrite of `text` via Groq.
    Rotates API keys on 429; only sleeps if ALL keys are exhausted.
    """
    instruction = (
        "Rewrite the following text at roughly half the length. "
        "Preserve all key facts and meaning. Return ONLY the rewritten text."
        if style == "short" else
        "Rewrite the following text at roughly 1.5× the length. "
        "Add relevant detail and explanation without inventing new facts. "
        "Return ONLY the rewritten text."
    )

    # Try every key in the pool before giving up
    keys_tried = 0
    while keys_tried < len(API_KEYS):
        for attempt in range(1, MAX_RETRIES_PER_KEY + 1):
            try:
                response = pool.current.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user",   "content": text},
                    ],
                    max_tokens=1024,
                    temperature=0.3,
                )
                time.sleep(0.4)   # gentle inter-call pacing
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                err_str = str(e)
                # Daily (TPD) limit — rotating key won't help until tomorrow
                if "tokens per day" in err_str.lower() or "tpd" in err_str.lower():
                    rotated = pool.rotate()
                    if rotated:
                        print(f"  [429 TPD] Daily limit on current key — switched.")
                        keys_tried += 1
                        break   # break inner loop, continue outer while
                    else:
                        # All keys hit daily limit — fall back to waiting
                        wait = _parse_wait_seconds(err_str)
                        print(f"\n  [429 TPD] All keys exhausted for today. "
                              f"Waiting {wait:.0f}s then retrying key[1]...")
                        pool.reset()
                        time.sleep(wait + 5)
                        keys_tried = 0
                        break
                else:
                    # Per-minute / per-request limit — wait briefly then retry same key
                    wait = min(_parse_wait_seconds(err_str), 65)
                    print(f"\n  [429 RPM] Rate limit — waiting {wait:.0f}s "
                          f"(attempt {attempt}/{MAX_RETRIES_PER_KEY})...")
                    time.sleep(wait + 2)

        else:
            # Inner for-loop exhausted retries on this key without success → rotate
            rotated = pool.rotate()
            if not rotated:
                raise RuntimeError("All Groq API keys exhausted. Try again tomorrow.")
            keys_tried += 1

    raise RuntimeError("generate_variant: could not complete after trying all keys.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_pairs(
    input_path:  str = "data/mt_bench.json",
    output_path: str = "data/pairs.jsonl",
    limit:       int = 80,
):
    """
    Read raw MT-Bench records, extract question + response pair,
    generate short/long variants, and save to output_path.

    - Resume-safe: skips pairs already in output_path.
    - Batched: prints progress every BATCH_SIZE pairs.
    """
    already_done = _count_existing(output_path)
    if already_done:
        print(f"  ↻ Resuming — {already_done} pairs already saved, skipping them.")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records  = records[:limit]
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

        print(f"  [{i+1}/{len(records)}] Generating variants...  ({pool.current_label})")
        pair = {
            "question": question,
            "short_a":  generate_variant(resp_a_raw, "short"),
            "long_a":   generate_variant(resp_a_raw, "long"),
            "short_b":  generate_variant(resp_b_raw, "short"),
            "long_b":   generate_variant(resp_b_raw, "long"),
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(pair) + "\n")

        new_pairs += 1

        if new_pairs % BATCH_SIZE == 0:
            total_so_far = already_done + new_pairs
            print(f"\n  ── Batch checkpoint: {total_so_far}/{limit} pairs saved ──\n")

    total = already_done + new_pairs
    print(f"\n✓ Done. {new_pairs} new pairs added — {total}/{limit} total in {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_pairs(
        input_path  = "data/mt_bench.json",
        output_path = "data/pairs.jsonl",
        limit       = 80,
    )