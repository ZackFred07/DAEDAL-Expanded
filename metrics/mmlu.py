
import json
import re
import os
import glob
from transformers import AutoTokenizer
import argparse

INDEX_TO_LETTER = ["A", "B", "C", "D"]
SPECIAL_TOKENS_PATTERN = re.compile(r"\[DONE\]|<\|eot_id\|>|<\|endoftext\|>")


def extract_answer_block(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_mmlu_letter_from_text(text: str) -> str | None:
    if text is None:
        return None

    answer_block = extract_answer_block(text)
    search_space = answer_block if answer_block else text

    letters = re.findall(r"\b([ABCD])\b", search_space.upper())
    if letters:
        return letters[-1]

    patterns = re.findall(
        r"(?i)answer\s*(?:is|:)?\s*\(?([ABCD])\)?",
        search_space,
    )
    if patterns:
        return patterns[-1].upper()

    letters = re.findall(r"\b([ABCD])\b", text.upper())
    if letters:
        return letters[-1]

    return None


def count_total_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_special_tokens(text: str) -> int:
    if not text:
        return 0
    return len(SPECIAL_TOKENS_PATTERN.findall(text))


def parse_mmlu_answers_from_jsonl(json_path: str, tokenizer):
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    total_correct = 0
    total_processed = 0
    total_raw_tokens_sum = 0
    total_special_tokens_sum = 0

    for item in data:
        total_processed += 1

        doc = item.get("doc", {})
        answer_idx = doc.get("answer", None)
        ground_truth_letter = None

        if answer_idx is not None:
            if isinstance(answer_idx, str):
                s = answer_idx.strip().upper()
                if s in ("A", "B", "C", "D"):
                    ground_truth_letter = s
                else:
                    try:
                        ground_truth_letter = INDEX_TO_LETTER[int(s)]
                    except (ValueError, IndexError):
                        ground_truth_letter = None
            else:
                try:
                    ground_truth_letter = INDEX_TO_LETTER[int(answer_idx)]
                except (ValueError, IndexError, TypeError):
                    ground_truth_letter = None

        if ground_truth_letter is None:
            target = str(item.get("target", "")).strip()
            last_char = target[-1:].upper()
            if last_char in INDEX_TO_LETTER:
                ground_truth_letter = last_char

        raw_generation = ""
        resps = item.get("resps")
        if (
            isinstance(resps, list)
            and resps
            and isinstance(resps[0], list)
            and resps[0]
        ):
            raw_generation = resps[0][0]

        total_raw_tokens = count_total_tokens(raw_generation, tokenizer)
        special_tokens = count_special_tokens(raw_generation)
        total_raw_tokens_sum += total_raw_tokens
        total_special_tokens_sum += special_tokens

        predicted_letter = extract_mmlu_letter_from_text(raw_generation)

        if (
            ground_truth_letter is not None
            and predicted_letter is not None
            and predicted_letter == ground_truth_letter
        ):
            total_correct += 1

    return (
        total_correct,
        total_processed,
        total_raw_tokens_sum,
        total_special_tokens_sum,
    )


def evaluate_mmlu_results(directory: str, tokenizer_path: str):
    print("\n" + "=" * 50 + f"\nProcessing directory: {directory}\n" + "=" * 50)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load Tokenizer, please check the path: {e}")
        return

    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return

    agg = {
        "correct": 0,
        "processed": 0,
        "total_raw_tokens": 0,
        "total_special_tokens": 0,
    }

    print(f"Found {len(jsonl_files)} files to process...")
    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        try:
            (
                correct,
                processed,
                raw_tokens,
                special_tokens,
            ) = parse_mmlu_answers_from_jsonl(json_path=file_path, tokenizer=tokenizer)

            agg["correct"] += correct
            agg["processed"] += processed
            agg["total_raw_tokens"] += raw_tokens
            agg["total_special_tokens"] += special_tokens

        except Exception as e:
            print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
            continue

    total_processed = agg["processed"]
    if total_processed == 0:
        print("No valid data processed. Cannot calculate results.")
        return

    accuracy = (agg["correct"] / total_processed) * 100.0

    total_raw_tokens = agg["total_raw_tokens"]
    total_special_tokens = agg["total_special_tokens"]
    avg_total_tokens = total_raw_tokens / total_processed if total_processed > 0 else 0.0
    effective_tokens = max(total_raw_tokens - total_special_tokens, 0)
    avg_effective_tokens = (
        effective_tokens / total_processed if total_processed > 0 else 0.0
    )
    e_ratio = (
        (effective_tokens / total_raw_tokens * 100.0)
        if total_raw_tokens > 0
        else 0.0
    )

    print("\n" + "-" * 80)
    print(f"Results for '{os.path.basename(directory)}'")
    print("-" * 80)
    print(f"  - Accuracy:                 {accuracy:.2f}%")
    print(f"  - Avg. Effective Tokens:    {avg_effective_tokens:.2f}")
    print(f"  - Avg. Total Tokens:        {avg_total_tokens:.2f}")
    print(f"  - Avg. Effective Token Ratio: {e_ratio:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the HF model/tokenizer (e.g., ./ckpts/LLaDA-1.5)",
    )
    parser.add_argument(
        "-r",
        "--res_path",
        type=str,
        required=True,
        help="Directory containing MMLU *.jsonl result files",
    )
    args = parser.parse_args()

    evaluate_mmlu_results(directory=args.res_path, tokenizer_path=args.model_path)
