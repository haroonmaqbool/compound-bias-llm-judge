# Replace the response-extraction block inside build_pairs():

for i, record in enumerate(records):
    question = record["conversation_a"][0]["content"]  # first user turn = the question

    # Last assistant message = the model's response
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

    print(f"  [{i+1}/{len(records)}] Generating variants...")
    pair = {
        "question": question,
        "short_a":  generate_variant(resp_a_raw, "short"),
        "long_a":   generate_variant(resp_a_raw, "long"),
        "short_b":  generate_variant(resp_b_raw, "short"),
        "long_b":   generate_variant(resp_b_raw, "long"),
    }
    pairs.append(pair)

    with open(output_path, "a") as f:
        f.write(json.dumps(pair) + "\n")