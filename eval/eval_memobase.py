import os
import re
import time
import json
import copy
import traceback

from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from llms import llm_request

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed
)

from memobase import MemoBaseClient, ChatBlob
import memobase.error
from prompts import PROMPT_MEMZERO

import pandas as pd
from sqlalchemy import create_engine, URL, inspect


load_dotenv()

RETRY_TIMES = 10
WAIT_TIME = 10

client = MemoBaseClient(
    project_url=os.getenv("MEMOBASE_PROJECT_URL"),
    api_key=os.getenv("MEMOBASE_PROJECT_TOKEN"),
)

DB_CONFIG = {
    "host": os.getenv("MEMOBASE_DB_HOST"),
    "port": os.getenv("MEMOBASE_DB_PORT"),
    "username": os.getenv("MEMOBASE_DB_USER"),
    "password": os.getenv("MEMOBASE_DB_PASSWORD"),
    "database": os.getenv("MEMOBASE_DB_NAME"),
    "limit": 100
}

TEMPLATE_MEMOBASE = """Memories for user {user_id}:
    {memories}
"""


TABLE_FIELD = {
    "user_profiles": "content",
    "user_events": "event_data"
}


def query_content_by_time(
    query_time: str,
    user_id: str,
    table_name: str = "user_events",
    host: str = DB_CONFIG["host"],
    port: int = DB_CONFIG["port"],
    username: str = DB_CONFIG["username"],
    password: str = DB_CONFIG["password"],
    database: str = DB_CONFIG["database"],
    limit: int = DB_CONFIG["limit"]
):
    
    if limit <= 0:
        raise ValueError("Limit must be a positive integer (>= 1)")
    
    try:
        query_time_utc = datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S")
        query_time_utc = query_time_utc.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError("Invalid time format. Use YYYY-MM-DD HH:MM:SS") from e
    
    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database
    )

    engine = create_engine(
        connection_url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30
    )

    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise RuntimeError(f"Table '{table_name}' does not exist in database '{database}'")
    
    field = TABLE_FIELD[table_name]
    query = f"""
         SELECT {field}
         FROM {table_name} 
         WHERE user_id = %s AND (created_at >= %s OR updated_at >= %s)
         ORDER BY created_at
         LIMIT {limit}
    """

    params = (user_id, query_time_utc, query_time_utc)
    
    try:
        df = pd.read_sql(query, engine, params=params).to_dict(orient="list")
        return df
    except Exception as e:
        print(f"Database query failed: {str(e)}: {traceback.format_exc()}")
        raise


def format_event_data(event_data: list):

    memories = []
    for item in event_data:
        item_memories = item["event_tip"].replace('```', "").strip().split("\n")
        
        for item in item.get("profile_delta", []):
            topic = item.get('attributes', {}).get('topic', '')
            sub_topic = item.get('attributes', {}).get('sub_topic', '')
            item_memories.append(f"{topic}::{sub_topic}::{item['content']}")

        memories.extend(item_memories)
    
    return memories


@retry(
    retry=retry_if_exception_type((Exception, memobase.error.ServerError)),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def add_peer_memory(client, user_id, messages):
    
    blob = ChatBlob(messages=messages)
    u = client.get_user(user_id)
    bid = u.insert(blob)
    u.flush(sync=True)


def add_memory(client, user_id, dialogues, batch_size=20):

    start = time.time() 
    start_add_time = (datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

    batch_size = 20
    for i in range(0, len(dialogues), batch_size):
        messages = [
            {
                "role": turn["role"],
                "content": turn["content"],
                "created_at": turn["timestamp"]
            } for turn in dialogues[i:i+batch_size]
        ]
        blob = ChatBlob(messages=messages)

        u = client.get_user(user_id)
        bid = u.insert(blob)
        u.flush(sync=True)

    end_add_time = (datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    memories = query_content_by_time(start_add_time, user_id)
    
    duration_ms = (time.time() - start) * 1000
    return memories, start_add_time, end_add_time, duration_ms


@retry(
    retry=retry_if_exception_type((Exception, memobase.error.ServerError)),
    wait=wait_fixed(WAIT_TIME),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True
)
def _search_memory(client, query, user_id, max_token_size=1000):
    start = time.time()
    
    u = client.get_user(user_id)
    context = u.context(
        max_token_size=max_token_size,
        chats=[{"role": "user", "content": query}],
        event_similarity_threshold=0.2,
        fill_window_with_events=True
    )

    memories = [item for item in context.split("\n") if item.startswith("- ")]
    
    duration_ms = (time.time() - start) * 1000
    return context, memories, duration_ms


def extract_user_name(persona_info: str):
    match = re.search(r'Name:\s*(.*?); Gender:', persona_info)

    if match:
        username = match.group(1).strip()
        return username
    else:
        raise ValueError("No name found.")
    

def search_memory(client, query, user_id, max_token_size=1000):
    start = time.time()
    try:
        return _search_memory(client, query, user_id, max_token_size)
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        print(f"[search_memory] All retries failed and returned the default value. Error message: {e}")
        return "", [], duration_ms
    

def process_user(user_data, max_token_size, save_path, version):

    user_name = extract_user_name(user_data["persona_info"]) + f"_{version}"
    user_id = client.add_user({
        "name":user_name
    })
    sessions = user_data["sessions"]
    
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")
    
    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "memobase_user_id": user_id,
        "sessions": []
    }

    try:

        for session in tqdm(sessions, total=len(sessions), desc=f"Processing user {user_name}"):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"]
            }

            # add messages
            dialogue = session["dialogue"]
            
            result, start_add_time, end_add_time, duration_ms = add_memory(
                client=client, 
                user_id=user_id, 
                dialogues=dialogue,
                batch_size=10
            )

            if session.get('is_generated_qa_session', False):
                new_session["add_dialogue_duration_ms"] = duration_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            memories = format_event_data(result.get("event_data", []))
            new_session["extracted_memories"] = memories
            new_session["add_dialogue_start_time"] = start_add_time
            new_session["add_dialogue_end_time"] = end_add_time
            new_session["add_dialogue_duration_ms"] = duration_ms

            # search updated memories
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory["original_memories"]:
                    continue

                _, memories_from_system, duration_ms = search_memory( 
                    client=client,
                    query=memory["memory_content"],
                    user_id=user_id,
                    max_token_size=250
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
                    max_token_size=max_token_size
                )
                
                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = duration_ms

                prompt = PROMPT_MEMZERO.format(
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
    max_token_size: int = 500,
    max_workers: int = 5
):
    frame = "memobase"
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
            future = executor.submit(process_user, user_data, max_token_size, save_path, version)
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
    max_token_size = 500

    main(
        data_path=data_path,
        version=version,
        max_token_size=max_token_size
    )