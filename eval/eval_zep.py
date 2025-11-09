import os
import re
import time
import uuid
import json
import copy
import traceback
from typing import Literal
from datetime import datetime, timezone
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from zep_cloud.client import Zep
from zep_cloud.types import Message
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed
)

from llms import llm_request
from prompts import PROMPT_ZEP


TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""


load_dotenv()

RETRY_TIMES = 10
WAIT_TIME = 60

client = Zep(
    api_key=os.getenv("ZEP_API_KEY")
)


def split_conversations(conversations, max_dialogues=28, max_chars=2400):

    result = []
    current_chunk = []
    current_char_count = 0

    for convo in conversations:
        content = convo['content']
        content_len = len(content)

        if content_len > max_chars:
            start = 0
            while start < content_len:
                end = start + max_chars
                sub_content = content[start:end]
                sub_convo = {**convo, 'content': sub_content}
                if (len(current_chunk) >= max_dialogues or
                    current_char_count + len(sub_content) > max_chars):
                    result.append(current_chunk)
                    current_chunk = []
                    current_char_count = 0
                current_chunk.append(sub_convo)
                current_char_count += len(sub_content)
                start = end
            continue

        if (len(current_chunk) >= max_dialogues or
            current_char_count + content_len > max_chars):
            result.append(current_chunk)
            current_chunk = []
            current_char_count = 0

        current_chunk.append(convo)
        current_char_count += content_len

    if current_chunk:
        result.append(current_chunk)

    return result


def split_and_clean(block):
    if not block:
        return []
    
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    return lines
    

def extract_facts_entities(text: str) -> list[str]:
    
    if any(keyword in text for keyword in [
        "FACTS and ENTITIES represent relevant context",
        "FACTS and ENTITIES, and EPISODES represent relevant context"
    ]):
        
        facts_match = re.search(r"<FACTS>(.*?)</FACTS>", text, re.DOTALL)
        entities_match = re.search(r"<ENTITIES>(.*?)</ENTITIES>", text, re.DOTALL)
        episodes_match = re.search(r"<EPISODES>(.*?)</EPISODES>", text, re.DOTALL)

        facts_list = split_and_clean(facts_match.group(1) if facts_match else "")
        entities_list = split_and_clean(entities_match.group(1) if entities_match else "")
        episodes_list = split_and_clean(episodes_match.group(1) if episodes_match else "")

        entities_list = [
            entity.strip()
            for entity in entities_list
            if entity.strip() and entity.strip().startswith("Summary")
        ]

        return facts_list + entities_list + episodes_list

    return split_and_clean(text)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def add_message(client, user_id, thread_id, message):
    date_format = "%b %d, %Y, %H:%M:%S"

    formatted_message = []
    for m in message:
        
        dt = datetime.strptime(m["timestamp"], date_format).replace(tzinfo=timezone.utc)
        iso_date = dt.isoformat()

        formatted_m = Message(
            name=user_id,
            role=m["role"],
            content=m["content"],
            created_at=iso_date
        )

        formatted_message.append(formatted_m)

    response = client.thread.add_messages(thread_id, messages=formatted_message)

    return response


def add_memory(client, user_id, thread_id, messages):

    start = time.time()

    client.thread.create(
        thread_id=thread_id,
        user_id=user_id,
    )

    splited_messages = split_conversations(messages)
    for message in splited_messages:
        
        response = add_message(client, user_id, thread_id, message)

    duration_ms = (time.time() - start) * 1000

    return duration_ms


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def get_thread_memory(client, thread_id):

    start = time.time()

    memory = client.thread.get_user_context(thread_id=thread_id, mode="basic")

    # Access the context block (for use in prompts)
    context_block = memory.context

    duration_ms = (time.time() - start) * 1000

    return extract_facts_entities(context_block), duration_ms


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def search_memory(
    client, 
    query, 
    user_id, 
    top_k=20
):
    start = time.time()

    edges_results = (
        client.graph.search(
            user_id=user_id, reranker="cross_encoder", query=query, scope="edges", limit=top_k//2
        )
    ).edges

    facts = [f'  - {edge.fact} (event_time: {edge.valid_at})' for edge in edges_results]

    node_results = (
        client.graph.search(
            user_id=user_id, reranker="rrf", query=query, scope="nodes", limit=top_k - top_k//2
        )
    ).nodes

    entities = [f'  - {node.name}: {node.summary}' for node in node_results]

    memories = facts + entities
    context = TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))

    duration_ms = (time.time() - start) * 1000

    return context, memories, duration_ms


def extract_user_name(persona_info: str):
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")


def process_user_add_memory(user_data, save_path, version):

    user_name = extract_user_name(user_data["persona_info"])
    user_id = user_name + "_" + version

    try:
        client.user.delete(user_name)
    except Exception as e:
        pass

    user = client.user.add(
        user_id=user_id,
        first_name=user_name.split()[0],
        last_name=user_name.split()[1],
    )

    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp-add")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "user_id": user_id,
        "sessions": []
    }

    try:

        for session in tqdm(sessions, total=len(sessions), desc=f"Processing (add) user {user_name}"):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"]
            }

            # add messages
            dialogue = session["dialogue"]
            thread_id = uuid.uuid4().hex
            
            duration_ms = add_memory(
                client=client, 
                user_id=user_id,
                thread_id=thread_id,
                messages=dialogue
            )

            new_session["zep_thread_id"] = thread_id

            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            new_session["add_dialogue_duration_ms"] = duration_ms

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = session["questions"]
        
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


def process_user_search_memory(user_data, top_k, save_path):

    user_name = user_data["user_name"]
    user_id = user_data["user_id"]

    sessions = user_data["sessions"]

    tmp_dir = os.path.join(save_path, "tmp-search")
    os.makedirs(tmp_dir, exist_ok=True)

    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "user_id": user_id,
        "sessions": []
    }

    try:

        for session in tqdm(sessions, total=len(sessions), desc=f"Processing (search) user {user_name}"):

            if session.get('is_generated_qa_session', False):
                new_user_data["sessions"].append(session)
                continue

            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"],
                "zep_thread_id": session["zep_thread_id"],
                "add_dialogue_duration_ms": session["add_dialogue_duration_ms"]
            }

            thread_id = session["zep_thread_id"]
            extracted_memories, duration_ms = get_thread_memory(client, thread_id)
            new_session["extracted_memories"] = extracted_memories

            # updated memories
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory["original_memories"]:
                    continue

                _, memories_from_system, duration_ms = search_memory(
                    client=client, 
                    query=memory["memory_content"][:395], 
                    user_id=user_id, 
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
                    client=client, 
                    query=qa["question"], 
                    user_id=user_id, 
                    top_k=top_k
                )

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = duration_ms

                prompt = PROMPT_ZEP.format(
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


def run_add(
    data_path: str,
    save_path: str,
    version: str = "default",
    max_workers: int = 2
):
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, user_data in enumerate(iter_jsonl(data_path), 1):
            
            uuid = user_data["uuid"]
            future = executor.submit(process_user_add_memory, user_data, save_path, version)
            futures[future] = uuid

        total_users = idx

        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] ✅ Finished (add) {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] ❌ Error processing (add) {uuid}: {e}")
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"✅All dialogues added! Finished in {elapsed:.2f}s")


def run_search(
    save_path: str,
    output_file: str,
    top_k: int = 20,
    max_workers: int = 2
):
    start_time = time.time()
    add_res_dir = os.path.join(save_path, "tmp-add")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, file_name in enumerate(os.listdir(add_res_dir), 1):
            if not file_name.endswith(".json"):
                continue
            user_file = os.path.join(add_res_dir, file_name)
            with open(user_file, "r", encoding="utf-8") as f:
                user_data = json.load(f)

            uuid = user_data["uuid"]
            future = executor.submit(process_user_search_memory, user_data, top_k, save_path)
            futures[future] = uuid

        total_users = idx
        
        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{total_users}] ✅ Finished (search) {uuid} ({result['status']})")
            except Exception as e:
                print(f"[{i}/{total_users}] ❌ Error processing (search) {uuid}: {e}")
                traceback.print_exc()
    
    search_res_dir = os.path.join(save_path, "tmp-search")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file in os.listdir(search_res_dir):
            if file.endswith(".json"):
                file_path = os.path.join(search_res_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        data = json.load(f_in)
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")

    elapsed = time.time() - start_time
    print(f"✅ All done in {elapsed:.2f}s")
    print(f"✅ Final results saved to: {output_file}")


def main(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    max_workers: int = 2,
    run_task: Literal['add', 'search'] = 'add'
):
    frame = "zep"
    save_path = f"results/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    
    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")

    if run_task == "add":
        run_add(data_path, save_path, version, max_workers)
    elif run_task == "search":
        run_search(save_path, output_file, top_k, max_workers)


if __name__ == "__main__":
    data_path = "../data/HaluMem-medium.jsonl"
    version = "default"
    top_k = 20
    run_task = "add"

    main(
        data_path=data_path,
        version=version,
        top_k=top_k,
        run_task=run_task
    )