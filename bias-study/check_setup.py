#!/usr/bin/env python
# check_setup.py
# ================================================================
# Run this BEFORE starting any experiments.
# Checks:
#   1. Python packages installed
#   2. Ollama running + 3 local models pulled
#   3. Groq API keys valid (tests each key, flags dead ones)
#   4. Data files present
#   5. Output directories exist
#
# Usage:  python check_setup.py        (run from bias-study/ folder)
# ================================================================

import os, sys, importlib, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ROOT = os.path.dirname(os.path.abspath(__file__))


def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


# ── 1. Python packages ────────────────────────────────────────────────────────
section("1. Python packages")

REQUIRED = ["requests", "groq", "dotenv", "pandas",
            "numpy", "matplotlib", "statsmodels", "scipy"]

all_ok = True
for pkg in REQUIRED:
    mod = "dotenv" if pkg == "dotenv" else pkg
    try:
        importlib.import_module(mod)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ MISSING: {pkg}  →  pip install {pkg}")
        all_ok = False

if not all_ok:
    print("\n  Install missing packages then re-run.\n")
    sys.exit(1)


# ── 2. Ollama ─────────────────────────────────────────────────────────────────
section("2. Ollama (local GPU models)")

import requests as req

OLLAMA_BASE   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = {
    "qwen2.5":   os.getenv("DEEPSEEK_MODEL",    "qwen2.5:14b"),
    "llama3-8b": os.getenv("MISTRAL_MODEL",     "llama3:8b"),
    "gemma2-9b": os.getenv("GROQ_VARIANT_MODEL","gemma2:9b"),
}

try:
    r = req.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    available = [m["name"] for m in r.json().get("models", [])]
    print(f"  ✓ Ollama running at {OLLAMA_BASE}")

    for judge_id, model_tag in OLLAMA_MODELS.items():
        found = any(model_tag in a or a.startswith(model_tag.split(":")[0])
                    for a in available)
        if found:
            print(f"  ✓ {judge_id:12s}  ({model_tag})")
        else:
            print(f"  ✗ {judge_id:12s}  ({model_tag}) — NOT PULLED")
            print(f"      Fix:  ollama pull {model_tag}")

except req.exceptions.ConnectionError:
    print(f"  ✗ Ollama NOT running at {OLLAMA_BASE}")
    print("  Start it:  ollama serve")
    for _, tag in OLLAMA_MODELS.items():
        print(f"  Then:      ollama pull {tag}")
except Exception as e:
    print(f"  ✗ Ollama error: {e}")


# ── 3. Groq API keys ──────────────────────────────────────────────────────────
section("3. Groq API keys")

from groq import Groq, AuthenticationError, RateLimitError

raw = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
keys = [k.strip() for k in raw.split(",") if k.strip()]

if not keys:
    print("  ✗ GROQ_API_KEYS not set in .env")
else:
    print(f"  Found {len(keys)} key(s). Testing each...\n")
    PROBE_MODEL = "llama-3.1-8b-instant"   # cheapest Groq model for probing
    working = []
    dead    = []

    for i, key in enumerate(keys):
        label = f"key[{i+1}]  ...{key[-8:]}"
        try:
            client = Groq(api_key=key)
            resp   = client.chat.completions.create(
                model=PROBE_MODEL,
                messages=[{"role":"user","content":"Reply A"}],
                max_tokens=3,
            )
            working.append(key)
            print(f"  ✓ {label}  — VALID")
        except AuthenticationError:
            dead.append(key)
            print(f"  ✗ {label}  — INVALID KEY (delete from .env)")
        except RateLimitError:
            working.append(key)   # key is valid, just rate-limited
            print(f"  ~ {label}  — RATE-LIMITED (key valid, quota hit)")
        except Exception as e:
            dead.append(key)
            print(f"  ✗ {label}  — ERROR: {e}")
        time.sleep(0.5)

    print(f"\n  Working keys: {len(working)}/{len(keys)}")
    if dead:
        print(f"  ✗ Dead keys  ({len(dead)}): remove these suffixes from .env:")
        for k in dead:
            print(f"      ...{k[-12:]}")

    # Check Groq judge models
    print()
    GROQ_MODELS = {
        "qwen3-32b":  os.getenv("GROQ_JUDGE_QWEN", "qwen/qwen3-32b"),
        "llama3-70b": os.getenv("GROQ_JUDGE_8B",   "llama-3.3-70b-versatile"),
    }
    if working:
        test_client = Groq(api_key=working[0])
        for judge_id, model_id in GROQ_MODELS.items():
            try:
                test_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role":"user","content":"Reply A"}],
                    max_tokens=3,
                )
                print(f"  ✓ {judge_id:12s}  ({model_id}) — accessible")
            except RateLimitError:
                print(f"  ~ {judge_id:12s}  ({model_id}) — rate-limited (model accessible)")
            except Exception as e:
                print(f"  ✗ {judge_id:12s}  ({model_id}) — {e}")
    else:
        print("  [SKIP] No working Groq keys — cannot probe models.")


# ── 4. Data files ─────────────────────────────────────────────────────────────
section("4. Data files")

DATA_FILES = {
    "data/mt_bench.json":        "Raw MT-Bench source",
    "data/pairs.jsonl":          "MT-Bench variants (80 pairs needed)",
    "data/summeval_pairs.jsonl": "SummEval pairs",
    "data/urdu_pairs.jsonl":     "Urdu translated pairs",
}

for rel, desc in DATA_FILES.items():
    full = os.path.join(ROOT, rel)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as fh:
            n = sum(1 for l in fh if l.strip())
        print(f"  ✓ {rel:35s} ({n} records)")
    else:
        print(f"  ✗ {rel:35s} MISSING — {desc}")


# ── 5. Output directories ─────────────────────────────────────────────────────
section("5. Output directories")

for d in ["results", "results/raw", "results/tables", "figures"]:
    full = os.path.join(ROOT, d)
    os.makedirs(full, exist_ok=True)
    print(f"  ✓ {d}/")


# ── Summary ───────────────────────────────────────────────────────────────────
section("Run Order (after all checks pass)")
print("""
  ┌───────────────────────────────────────────────────────────┐
  │  Run these in order from the bias-study/ directory        │
  │                                                           │
  │  Step 0 — One-time Ollama setup (separate terminal)       │
  │    ollama serve                                           │
  │    ollama pull qwen2.5:14b                                │
  │    ollama pull llama3:8b                                  │
  │    ollama pull gemma2:9b                                  │
  │                                                           │
  │  Step 1 — Generate MT-Bench variants (Ollama local)       │
  │    python src/variant_generator.py                        │
  │                                                           │
  │  Step 2 — MT-Bench English experiment                     │
  │    python src/experiment.py                               │
  │                                                           │
  │  Step 3 — SummEval experiment                             │
  │    python src/run_summeval_experiment.py                  │
  │                                                           │
  │  Step 4 — Urdu experiment                                 │
  │    python src/run_urdu_experiment.py                      │
  │                                                           │
  │  Step 5 — Consolidate & clean                             │
  │    python src/consolidate.py                              │
  │                                                           │
  │  Step 6 — Statistical analysis                            │
  │    python src/analysis_v2.py                              │
  │                                                           │
  │  Step 7 — Generate figures                                │
  │    python src/figures.py                                  │
  └───────────────────────────────────────────────────────────┘
""")
