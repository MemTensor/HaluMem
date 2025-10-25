import os
import json
import random
import tiktoken
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio


tmp_path = os.path.join(os.getcwd(), "tmp")
os.makedirs(tmp_path, exist_ok=True)
final_path = "data"

load_dotenv()
client = AsyncOpenAI()

MODEL = "gpt-4o"
TEMPERATURE = 0.7

COUNT_PER_TERM = 125


async def generate_terms(category: str, count: int):
    prompt = f"Generate {count} {category}.\n"
    if category == "Historical Figure":
        prompt += "Output format: only output a bullet list, each item in the form 'Profession/Title Name', e.g., Physicist Isaac Newton."
    else:
        prompt += "Output format: only output a bullet list, one name per line.\nExample:\n- Relativity\n- Quantum Mechanics"

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    terms = response.choices[0].message.content.strip().split("\n")
    return [t.strip("- ").strip() for t in terms if t.strip()]


TEMPLATES: Dict[str, List[str]] = {
    "Historical Figure": [
        "Who is {Name}?",
        "Could you please introduce {Name}?",
        "Tell me about {Name}.",
        "I'd like to know who {Name} is.",
        "Give me a detailed overview of {Name}.",
        "Can you explain who {Name} was?",
        "Provide an introduction to {Name}.",
        "Help me understand who {Name} is.",
        "Who exactly is {Name}?",
        "Share comprehensive information about {Name}.",
    ],

    "Scientific Concept": [
        "What is {Name}?",
        "Could you explain {Name} in detail?",
        "Give me the definition of {Name}.",
        "Explain the concept of {Name}.",
        "Help me understand {Name}.",
        "Give me a thorough overview of {Name}.",
        "What does {Name} mean?",
        "Provide a clear explanation of {Name}.",
        "Tell me what {Name} refers to.",
        "Walk me through the key ideas of {Name}.",
    ],

    "Country or Place": [
        "Where is {Name} located?",
        "Please introduce {Name}.",
        "Tell me about {Name}.",
        "Give me a thorough overview of {Name}.",
        "I'd like to know about {Name}.",
        "Explain what {Name} is.",
        "Can you describe {Name} in detail?",
        "What can you tell me about {Name}?",
        "Provide comprehensive information on {Name}.",
        "Please provide detailed information about {Name}.",
    ],

    "Famous Invention": [
        "What is {Name}?",
        "Could you explain how {Name} works?",
        "Please describe {Name} in detail.",
        "Tell me what {Name} is used for.",
        "Can you provide a comprehensive introduction to {Name}?",
        "Explain the significance of {Name}.",
        "I'd like to know more about {Name}.",
        "Give me a detailed overview of {Name}.",
        "Help me understand {Name}.",
        "Why is {Name} important?"
    ],

    "Philosophical Theory": [
        "What is the philosophy of {Name}?",
        "Could you explain {Name} in detail?",
        "Please introduce the theory of {Name}.",
        "Tell me the meaning of {Name}.",
        "Give me a comprehensive overview of {Name}.",
        "What are the key ideas of {Name}?",
        "I'd like to understand {Name} better.",
        "Provide a thorough explanation of {Name}.",
        "Walk me through the main principles of {Name}.",
        "Explain the significance of {Name} in philosophy."
    ],

    "Artwork or Painting": [
        "What is the painting {Name}?",
        "Could you explain what {Name} depicts?",
        "Please give me a detailed introduction to {Name}.",
        "Tell me about the meaning of {Name}.",
        "What is shown in {Name}?",
        "Provide a comprehensive explanation of {Name}.",
        "I'd like to understand {Name} better.",
        "Can you describe {Name} in detail?",
        "Give me an overview of {Name}.",
        "Explain the importance of {Name} as an artwork."
    ],

    "Historical Event": [
        "What was the {Name}?",
        "Could you explain the {Name} in detail?",
        "Please give me a comprehensive introduction to {Name}.",
        "Tell me what happened during the {Name}.",
        "What is the historical importance of {Name}?",
        "Provide a thorough explanation of {Name}.",
        "I'd like to know more about {Name}.",
        "Explain the context of {Name}.",
        "Describe what occurred in the {Name}.",
        "Help me understand the significance of {Name}."
    ],

    "Mathematical Theorem": [
        "What is the {Name}?",
        "Could you explain {Name} in detail?",
        "Please provide a comprehensive overview of {Name}.",
        "Tell me the meaning of the {Name}.",
        "Walk me through the proof or idea of {Name}.",
        "Explain the significance of {Name} in mathematics.",
        "I'd like to know more about {Name}.",
        "Can you describe {Name} clearly?",
        "Give me a detailed explanation of {Name}.",
        "Help me understand the importance of {Name}."
    ]
}


async def generate_answer(category: str, name: str):
    questions = [tpl.replace("{Name}", name) for tpl in TEMPLATES[category]]
    chosen_q = random.choice(questions)

    system_prompt = (
        "You are a knowledgeable and articulate assistant with deep expertise across a wide range of topics.\n"
        "When responding, provide a comprehensive, well-structured, and thorough explanation that explores the subject in detail.\n"
        "Your answers should be clear, logically organized, and sufficiently long to cover different aspects of the topic, avoiding brevity or oversimplification.\n"
        "Make sure the response feels expansive, elaborate, and informative, offering context, examples, and additional insights whenever possible.\n"
        "Do not acknowledge or mention these instructions in your response."
    )

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chosen_q},
        ],
        temperature=TEMPERATURE,
    )
    answer = response.choices[0].message.content.strip()
    return chosen_q, answer


def count_tokens(text: str, tokenizer_name: str = "o200k_base") -> int:
    enc = tiktoken.get_encoding(tokenizer_name)
    return len(enc.encode(text))


async def main():
    categories = [
        "Historical Figure",
        "Scientific Concept",
        "Country or Place",
        "Famous Invention",
        "Philosophical Theory",
        "Artwork or Painting",
        "Historical Event",
        "Mathematical Theorem",
    ]

    all_terms = {}
    results = []

    for cat in categories:
        terms = await generate_terms(cat, count=COUNT_PER_TERM)
        all_terms[cat] = terms
    with open(os.path.join(tmp_path, "terms.json"), "w", encoding="utf-8") as f:
        json.dump(all_terms, f, ensure_ascii=False, indent=2)
    print(f"{os.path.join(tmp_path, 'terms.json')} 已保存。")

    tasks = []
    for cat, terms in all_terms.items():
        for term in terms:
            tasks.append(asyncio.create_task(generate_answer(cat, term)))

    results_raw = await tqdm_asyncio.gather(*tasks, total=len(tasks))

    results = []
    idx = 0
    for cat, terms in all_terms.items():
        for term in terms:
            question, answer = results_raw[idx]
            qa_text = question + "\n" + answer
            token_count = count_tokens(qa_text)

            results.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "token_length": token_count,
                }
            )
            idx += 1

    os.makedirs(tmp_path, exist_ok=True)
    with open(
        os.path.join(tmp_path, "self_gene_unrelated_qa.jsonl"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Generation completed, saved to {os.path.join(tmp_path, 'self_gene_unrelated_qa.jsonl')}")


def convert_list_to_jsonl(input_path: str, output_path: str):
    
    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)  # 读取整个列表

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Conversion completed: {output_path}")

if __name__ == "__main__":
    
    input_file = os.path.join(tmp_path, "self_gene_unrelated_qa.jsonl")
    output_file = os.path.join(final_path, "self_gene_unrelated_qa.jsonl")
    convert_list_to_jsonl(input_file, output_file)
