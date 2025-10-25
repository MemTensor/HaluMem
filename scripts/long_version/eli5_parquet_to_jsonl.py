import argparse
import json
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import tiktoken


DEFAULT_INPUT_PARQUET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "data",
    "eli5.parquet",
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "data",
    "eli5_messages.jsonl",
)


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
        row.get("question_title"),
        row.get("title"),
        row.get("question_body"),
        row.get("prompt"),
        row.get("input"),
    ]
    return _first_non_empty_string(candidates)


def extract_answer(row: Dict[str, Any]) -> Optional[str]:
    
    candidates = [
        row.get("answer"),
        row.get("true_answer"),
        row.get("best_answer"),
        row.get("reference_answer"),
        row.get("response"),
        row.get("output"),
        row.get("answers"),  
        row.get("outputs"),   
    ]
    return _first_non_empty_string(candidates)


def row_to_messages(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = extract_question(row)
    a = extract_answer(row)
    if not q or not a:
        return None
    
    q = preprocess_text(q)
    a = preprocess_text(a)
    
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]
    return {"messages": messages}


def load_tokenizer(tokenizer_name: str = "o200k_base"):
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        raise ValueError(f"Failed to load the tiktoken tokenizer {tokenizer_name}: {e}")


def _count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


URL_PLACEHOLDER_RE = re.compile(r"\b_?URL_\d+_?\b", re.IGNORECASE)
PAREN_URL_PLACEHOLDER_RE = re.compile(r"\(\s*_?URL_\d+_?\s*\)")
MD_LINK_PLACEHOLDER_RE = re.compile(r"\[([^\]]+)\]\((_?URL_\d+_?)\)")
MULTISPACE_RE = re.compile(r"[ \t\f\v]+")


def _strip_control_chars(text: str) -> str:
    
    return "".join(
        ch for ch in text
        if (ch in "\n\r\t") or not unicodedata.category(ch).startswith("C") or ch == "\u200d"
    )


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text
    s = MD_LINK_PLACEHOLDER_RE.sub(lambda m: m.group(1), s)
    s = PAREN_URL_PLACEHOLDER_RE.sub("", s)
    s = URL_PLACEHOLDER_RE.sub("", s)
    s = strip_excessive_combining_marks(s, threshold=10)
    s = _strip_control_chars(s)
    s = MULTISPACE_RE.sub(" ", s)
    s = s.strip()
    return s


def strip_excessive_combining_marks(text: str, threshold: int = 10) -> str:

    mn_count = sum(1 for ch in text if unicodedata.category(ch) == 'Mn')
    if mn_count > threshold:
        return "".join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text


def convert_parquet_to_jsonl(input_parquet: str, output_jsonl: str, top_k: int = 10000) -> None:
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"The input parquet does not exist: {input_parquet}")

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    df = pd.read_parquet(input_parquet)

    tokenizer = load_tokenizer("o200k_base")

    collected = []
    for idx in range(len(df)):
        row_dict: Dict[str, Any] = df.iloc[idx].to_dict()
        obj = row_to_messages(row_dict)
        if obj is None:
            continue
        try:
            q_text = obj["messages"][0]["content"]
            a_text = obj["messages"][1]["content"]
            token_length = _count_tokens(q_text, tokenizer) + _count_tokens(a_text, tokenizer)
        except Exception:
            token_length = 0
        collected.append((token_length, obj["messages"]))

    collected.sort(key=lambda x: x[0], reverse=True)
    selected = collected[: max(0, int(top_k))]

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for token_length, messages in selected:
            record = {
                "messages": messages,
                "token_length": token_length,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Completed: Read {len(df)} records from {input_parquet}, selected the top-{len(selected)} records by token length, and wrote them to {output_jsonl}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ELI5/Universal QA Parquet to Messages format JSONL"
    )
    parser.add_argument(
        "--input",
        dest="input_parquet",
        default=DEFAULT_INPUT_PARQUET,
        help="Enter the parquet path (default: project data/eli5.parquet)",
    )
    parser.add_argument(
        "--output",
        dest="output_jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path (default: project data/eli5_cessages.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="Select the top K in descending order of token length (default: 10000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_parquet_to_jsonl(
        input_parquet=os.path.abspath(args.input_parquet),
        output_jsonl=os.path.abspath(args.output_jsonl),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()


