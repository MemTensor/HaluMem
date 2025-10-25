import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken


DEFAULT_INPUT_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "data",
    "native_math_gpt-oss-120b.jsonl",
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "data",
    "native_math_messages_sample.jsonl",
)


def load_tokenizer(tokenizer_name: str = "o200k_base"):
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        raise ValueError(f"Failed to load the tiktoken tokenizer {tokenizer_name}: {e}")


def _first_non_empty_string(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    for k in [
                        "text",
                        "answer",
                        "output",
                        "content",
                        "body",
                        "best_answer",
                    ]:
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
        if isinstance(value, dict):
            for k in [
                "text",
                "answer",
                "output",
                "content",
                "body",
                "best_answer",
            ]:
                v = value.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return None


def extract_question(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("question"),
        row.get("Question"),
        row.get("prompt"),
        row.get("Prompt"),
        row.get("input"),
        row.get("Input"),
        row.get("problem"),
        row.get("Problem"),
        row.get("query"),
        row.get("Query"),
        row.get("title"),
        row.get("Title"),
    ]
    return _first_non_empty_string(candidates)


def extract_answer(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("answer"),
        row.get("Answer"),
        row.get("Anwser"),
        row.get("solution"),
        row.get("Solution"),
        row.get("response"),
        row.get("Response"),
        row.get("output"),
        row.get("Output"),
        row.get("reference_answer"),
        row.get("Reference_Answer"),
        row.get("answers"),
        row.get("Answers"),
        row.get("outputs"),
        row.get("Outputs"),
        row.get("CoT_Native_Reasoning"),
    ]
    return _first_non_empty_string(candidates)


def _count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def row_to_record(row: Dict[str, Any], tokenizer) -> Optional[Dict[str, Any]]:
    q = extract_question(row)
    a = extract_answer(row)
    if not q or not a:
        return None
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    token_length = _count_tokens(q, tokenizer) + _count_tokens(a, tokenizer)
    return {"messages": messages, "token_length": token_length}


def reservoir_sample(iterable: Iterable[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    sample: List[Dict[str, Any]] = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


QUESTION_PREFIXES: List[str] = [
    "Hi, I have a question: ",
    "I'm stuck on this problem—could you help? ",
    "How would you approach this? ",
    "Could we reason through this together: ",
    "I tried a few ideas but failed. What's the right approach? ",
    "Starting from basics, how should I solve this? ",
    "It seems simple, but I'm unsure about the details: ",
    "Could you walk me through it step by step? ",
    "I'd like to verify my understanding—how do we solve this: ",
    "If you were not looking at a solution, how would you begin? ",
    "Could you help me make sense of this problem? ",
    "What would be a clean way to solve this? ",
    "Before I overcomplicate it, what's the key insight? ",
    "Can you outline a strategy for this question? ",
    "Where should I start to avoid common pitfalls? ",
]

ANSWER_PREFIXES: List[str] = [
    "Sure, let's start with the idea: ",
    "Alright, here's a step-by-step explanation: ",
    "Here's my approach with key derivations: ",
    "First the conclusion, then the reasoning: ",
    "Here's a clear solution: ",
    "Let's outline the key steps: ",
    "A concise solution goes like this: ",
    "We can break it down as follows: ",
    "Here's the core idea and its execution: ",
    "Step by step, this works: ",
]

def apply_dialogue_prefixes(record: Dict[str, Any], rng: random.Random, enable: bool) -> Dict[str, Any]:
    if not enable:
        return record
    try:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            return record
        user_msg = messages[0]
        asst_msg = messages[1]
        if not (isinstance(user_msg, dict) and isinstance(asst_msg, dict)):
            return record
        if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
            return record
        q_prefix = rng.choice(QUESTION_PREFIXES)
        a_prefix = rng.choice(ANSWER_PREFIXES)
        user_msg["content"] = f"{q_prefix}{user_msg['content']}"
        asst_msg["content"] = f"{a_prefix}{asst_msg['content']}"
        record["messages"] = [user_msg, asst_msg]
        return record
    except Exception:
        return record


def convert_jsonl_random_sample(
    input_jsonl: str,
    output_jsonl: str,
    sample_size: int = 10000,
    seed: Optional[int] = None,
    enable_prefix: bool = True,
) -> None:
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"The input jsonl file does not exist: {input_jsonl}")

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    tokenizer = load_tokenizer("o200k_base")
    rng = random.Random(seed)

    def valid_records() -> Iterable[Dict[str, Any]]:
        with open(input_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rec = row_to_record(row, tokenizer)
                if rec is not None:
                    yield rec

    # 进行水塘采样
    sampled = reservoir_sample(valid_records(), max(0, int(sample_size)), rng)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for rec in sampled:
            rec = apply_dialogue_prefixes(rec, rng, enable_prefix)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Completed: Randomly sample {len(sampled)} from {input_jsonl} and write them to {output_jsonl}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample from native_math JSONL and convert to messages+token_length format"
    )
    parser.add_argument(
        "--input",
        dest="input_jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Enter the JSON path (default: project data/native_math_gpt-oss-120b.jsonl)",
    )
    parser.add_argument(
        "--output",
        dest="output_jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path (default: project data/native_math_cessages_sample.jsonl)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Random sample size (default: 3000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "--disable-prefix",
        action="store_true",
        help="Close the dialogue prefix template (default enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_jsonl_random_sample(
        input_jsonl=os.path.abspath(args.input_jsonl),
        output_jsonl=os.path.abspath(args.output_jsonl),
        sample_size=args.sample_size,
        seed=args.seed,
        enable_prefix=(not args.disable_prefix),
    )


if __name__ == "__main__":
    main()


