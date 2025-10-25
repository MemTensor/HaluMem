import os
import re
import time
import uuid
import json
import copy
import requests
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed
)

from llms import llm_request
from prompts import PROMPT_MEMOS


TEMPLATE_MEMOS = """Memories for user {user_id}:

    {memories}
"""


load_dotenv()

RETRY_TIMES = 10
WAIT_TIME = 60

memos_url = os.getenv("MEMOS_URL")
headers = {"Content-Type": "application/json", "Authorization": os.getenv("MEMOS_KEY")}


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def add(messages, user_id, conv_id):

    start = time.time()

    url = f"{memos_url}/product/add"
    payload = json.dumps({"messages": messages, "user_id": user_id,
                            "mem_cube_id": user_id, "conversation_id": conv_id})
    response = requests.request("POST", url, data=payload, headers=headers)
    response_json = json.loads(response.text)
    assert response.status_code == 200, response.text
    assert response_json["message"] == 'Memory added successfully', response.text

    duration_ms = (time.time() - start) * 1000

    return response_json, duration_ms


def add_dialogue(dialogue, user_id, conv_id):
    """Add a dialogue to Memos."""

    date_format = "%b %d, %Y, %H:%M:%S"

    formatted_dialogue = [
        {
            "role": turn["role"],
            "content": turn["content"],
            "chat_time": datetime.strptime(turn["timestamp"], date_format).replace(tzinfo=timezone.utc).isoformat(),
        }
        for turn in dialogue
    ]

    batch_num = 20
    memories = []
    duration_ms = 0

    for i in range(0, len(formatted_dialogue), batch_num):
        batch = formatted_dialogue[i:i+batch_num]
        response, batch_duration_ms = add(batch, user_id, conv_id)
        memories.extend(
            [item['memory'] for item in response["data"]]
        )
        duration_ms += batch_duration_ms

    return memories, duration_ms


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def search_memory(query, user_id, top_k):
    """Search memories."""

    start = time.time()

    url = f"{memos_url}/product/search"
    payload = json.dumps(
        {"query": query, "user_id": user_id, "mem_cube_id": user_id,
            "conversation_id": "", "top_k": top_k, }, ensure_ascii=False)
    response = requests.request("POST", url, data=payload, headers=headers)
    assert response.status_code == 200, response.text
    assert json.loads(response.text)["message"] == "Search completed successfully", response.text

    results = json.loads(response.text)["data"]
    memories = [i["memory"] for i in results["text_mem"][0]["memories"]]

    context = TEMPLATE_MEMOS.format(
        user_id=user_id,
        memories="\n".join(memories)
    )

    duration_ms = (time.time() - start) * 1000

    return context, memories, duration_ms


def extract_user_name(persona_info: str):
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")
    

def process_user(user_data, top_k, save_path, version):

    user_name = extract_user_name(user_data["persona_info"]) + f"_{version}"
    print("user_name: ", user_name)
    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": []
    }

    try:
        session_id = 0

        for session in tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}"):

            session_id += 1
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"]
            }

            # add messages
            dialogue = session["dialogue"]

            conv_id = f"{session_id}_{user_name}"
            result, duration_ms = add_dialogue(
                dialogue=dialogue, 
                user_id=user_name, 
                conv_id=conv_id
            )

            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            extracted_memories = result
            new_session["extracted_memories"] = extracted_memories
            new_session["add_dialogue_duration_ms"] = duration_ms
            new_session["conv_id"] = conv_id

            # search updated memories
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory["original_memories"]:
                    continue

                _, memories_from_system, duration_ms = search_memory( 
                    query=memory["memory_content"],
                    user_id=user_name, 
                    top_k=10
                )

                memory["memories_from_system"] = memories_from_system

            # search and query
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []

            for qa in session["questions"]:

                context, _, duration_ms = search_memory(
                    query=qa["question"], 
                    user_id=user_name, 
                    top_k=top_k
                )

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = duration_ms

                prompt = PROMPT_MEMOS.format(
                    context=context,
                    question=qa["question"]
                )

                start_time = time.time()
                response = llm_request(prompt)
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = (time.time() - start_time) * 1000

                new_session["questions"].append(new_qa)
            
            new_user_data["sessions"].append(new_session)
        
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False)

        print(f"✅ Saved user {user_name} to {tmp_file}")
        return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}

    except Exception as e:
        error_path = os.path.join(tmp_dir, f"{user_data['uuid']}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Error in user {user_name}: {e}")
        return {"uuid": user_data["uuid"], "status": "error", "path": error_path}


def iter_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    max_workers: int = 2
):
    frame = "memos"
    save_path = f"results/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, user_data in enumerate(iter_jsonl(data_path), 1):
            uuid = user_data["uuid"]
            future = executor.submit(process_user, user_data, top_k, save_path, version)
            futures[future] = uuid

        total_users = idx

        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] ✅ Finished {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] ❌ Error processing {uuid}: {e}")
                traceback.print_exc()

    with open(output_file, "a", encoding="utf-8") as f_out:
        for file in os.listdir(tmp_dir):
            if file.endswith(".json"):
                file_path = os.path.join(tmp_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")

    elapsed = time.time() - start_time
    print(f"✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")


if __name__ == "__main__":
    data_path = "../data/HaluMem-medium.jsonl"
    version = "default"
    top_k = 20

    main(
        data_path=data_path,
        version=version,
        top_k=top_k
    )