import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

 
# ---------- Supplements ----------

def load_supplement(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages = obj.get("messages")
                tl = int(obj.get("token_length", 0))
                if isinstance(messages, list) and len(messages) >= 2:
                    items.append({"messages": messages, "token_length": max(1, tl)})
            except Exception:
                continue
    return items


def build_weighted_index(items: List[Dict[str, Any]], weighting: str) -> Tuple[List[int], List[float]]:
    idx = list(range(len(items)))
    if weighting == "token_length":
        w = [float(max(1, it.get("token_length", 1))) for it in items]
    else:
        w = [1.0 for _ in items]
    s = sum(w)
    if s <= 0:
        w = [1.0 for _ in items]
        s = sum(w)
    w = [wi / s for wi in w]
    return idx, w


def weighted_choice(idx: List[int], w: List[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in zip(idx, w):
        acc += p
        if r <= acc:
            return i
    return idx[-1]


# ---------- Insertion helpers ----------

DT_FMT = "%b %d, %Y, %H:%M:%S"  # matches like "Aug 23, 2025, 09:00:00"


def parse_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, DT_FMT)
    except Exception:
        return None


def format_dt(dt: datetime) -> str:
    return dt.strftime(DT_FMT)


def pick_random_time_between(t0: Optional[str], t1: Optional[str], rng: random.Random) -> Optional[str]:
    d0 = parse_dt(t0) if t0 else None
    d1 = parse_dt(t1) if t1 else None
    if d0 and d1 and d1 > d0:
        delta = (d1 - d0).total_seconds()
        r = rng.random()
        return format_dt(d0 + timedelta(seconds=delta * r))
    if d0:
        return format_dt(d0)
    if d1:
        return format_dt(d1)
    raise ValueError("Cannot determine time between sessions; both boundaries missing")


def reindex_dialogue_turns(dialogue: List[Dict[str, Any]]):
    # dialogue is flat messages; group by pairs in order, then rewrite turns 0..n-1
    # Assume messages are already in alternating roles and grouped in pairs (user, assistant)
    turn = -1
    last_role = None
    for msg in dialogue:
        role = msg.get("role")
        if role == "user":
            turn += 1
        msg["dialogue_turn"] = turn
        last_role = role


def split_into_turns(dialogue: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for msg in dialogue:
        if msg.get("role") == "user":
            # start new turn
            if current:
                turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    # ensure each turn has at most two messages
    normalized: List[List[Dict[str, Any]]] = []
    for t in turns:
        if len(t) >= 2:
            normalized.append([t[0], t[1]])
        else:
            normalized.append(t)
    return normalized


def flatten_turns(turns: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for t in turns:
        for m in t:
            flat.append(m)
    return flat


def insert_internal(session: Dict[str, Any], qa_messages: List[Dict[str, Any]], qa_token_len: int, rng: random.Random):
    # Insert a full QA at a random turn boundary (0..num_turns)
    dialogue: List[Dict[str, Any]] = session.get("dialogue", [])
    turns_list = split_into_turns(dialogue)
    num_turns = len(turns_list)
    insert_pos = rng.randint(0, num_turns)  # inclusive end means append when == num_turns

    # choose timestamp for inserted turn: prefer previous turn's last ts; else next; else None
    ts = None
    if insert_pos > 0 and turns_list[insert_pos - 1]:
        ts = turns_list[insert_pos - 1][-1].get("timestamp")
    elif insert_pos < num_turns and turns_list[insert_pos]:
        ts = turns_list[insert_pos][0].get("timestamp")

    new_turn = []
    for m in qa_messages:
        nm = {"role": m.get("role"), "content": m.get("content")}
        if ts:
            nm["timestamp"] = ts
        new_turn.append(nm)

    turns_list[insert_pos:insert_pos] = [new_turn]

    # flatten and reindex
    session["dialogue"] = flatten_turns(turns_list)
    reindex_dialogue_turns(session["dialogue"])
    session["dialogue_turn_num"] = len(turns_list)
    # increment per-session token length by inserted QA tokens
    old_len = int(session.get("dialogue_token_length") or 0)
    session["dialogue_token_length"] = old_len + int(qa_token_len)


def create_new_session(existing_sessions: List[Dict[str, Any]], qa_batch: List[List[Dict[str, Any]]], qa_token_sum: int, new_session_title: str, rng: random.Random) -> Dict[str, Any]:
    # Determine time gap: randomly pick adjacent sessions and sample a timestamp in-between
    start_time = None
    end_time = None
    if existing_sessions:
        gaps: List[Tuple[Optional[str], Optional[str]]] = []
        for i in range(len(existing_sessions) - 1):
            t0 = existing_sessions[i].get("end_time")
            t1 = existing_sessions[i + 1].get("start_time")
            gaps.append((t0, t1))
        if gaps:
            t0, t1 = rng.choice(gaps)
            sampled = pick_random_time_between(t0, t1, rng)
            start_time = sampled
            end_time = sampled
        else:
            # only one session; sample around its end_time
            t0 = existing_sessions[-1].get("end_time")
            sampled = pick_random_time_between(t0, t0, rng)
            start_time = sampled
            end_time = sampled
    # Build dialogue flat list
    dialogue: List[Dict[str, Any]] = []
    for qa in qa_batch:
        for m in qa:
            nm = {"role": m.get("role"), "content": m.get("content")}
            if start_time:
                nm["timestamp"] = start_time
            dialogue.append(nm)
    session = {
        "start_time": start_time,
        "end_time": end_time,
        "title": new_session_title,
        "memory_points_count": 0,
        "memory_points": [],
        "dialogue_turn_num": 0,
        "dialogue_token_length": int(qa_token_sum),
        "dialogue": dialogue,
        "questions": [],
        "question_count": 0,
    }
    reindex_dialogue_turns(session["dialogue"])
    turns = 0
    if session["dialogue"]:
        turns = session["dialogue"][-1].get("dialogue_turn", 0) + 1
    session["dialogue_turn_num"] = turns
    return session


def main():
    
    with open('config.json', 'r', encoding='utf-8') as f:
        cfg_root = json.load(f)
        cfg = cfg_root['STAGE6']['substage3_long']

    file_paths = cfg['file_paths']
    sampling = cfg['sampling']
    insertion = cfg['insertion']
    validation = cfg['validation']

    ratio = float(insertion.get('new_session_ratio', 0.5))
    
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("new_session_ratio must be between 0.0 and 1.0")

    original_output = file_paths['output_file']
    base_name, ext = os.path.splitext(original_output)
    file_paths['output_file'] = f"{base_name}_ratio{ratio:.1f}{ext}"

    rng = random.Random(sampling.get('seed'))

    supp_native = load_supplement(file_paths['supplement_native_math'])
    supp_eli5 = load_supplement(file_paths['supplement_eli5'])
    supp_self_gene = load_supplement(file_paths['supplement_self_gene'])
    if not supp_native and not supp_eli5 and not supp_self_gene:
        raise RuntimeError("No supplement data loaded.")

    idx_native, w_native = build_weighted_index(supp_native, sampling.get('weighting', 'token_length')) if supp_native else ([], [])
    idx_eli5, w_eli5 = build_weighted_index(supp_eli5, sampling.get('weighting', 'token_length')) if supp_eli5 else ([], [])
    idx_self_gene, w_self_gene = build_weighted_index(supp_self_gene, sampling.get('weighting', 'token_length')) if supp_self_gene else ([], [])

    src_dist = sampling.get('source_distribution', {"native_math": 0.33, "eli5": 0.33, "self_gene": 0.34})
    p_native = float(src_dist.get("native_math", 0.33))
    p_eli5 = float(src_dist.get("eli5", 0.33))
    p_self_gene = float(src_dist.get("self_gene", 0.34))
    total_p = p_native + p_eli5 + p_self_gene
    if total_p <= 0:
        p_native = 0.33
        p_eli5 = 0.33
        p_self_gene = 0.34
        total_p = 1.0
    p_native /= total_p
    p_eli5 /= total_p
    p_self_gene /= total_p

    def draw_one() -> Optional[Dict[str, Any]]:
        # choose source then choose item
        r = rng.random()
        if r < p_native and supp_native:
            i = weighted_choice(idx_native, w_native, rng)
            return supp_native[i]
        elif r < p_native + p_eli5 and supp_eli5:
            i = weighted_choice(idx_eli5, w_eli5, rng)
            return supp_eli5[i]
        elif supp_self_gene:
            i = weighted_choice(idx_self_gene, w_self_gene, rng)
            return supp_self_gene[i]
        raise ValueError("Supplement sources exhausted or unavailable; cannot draw sample")

    os.makedirs(os.path.dirname(os.path.abspath(file_paths['output_file'])), exist_ok=True)

    written = 0
    with open(file_paths['input_file'], "r", encoding="utf-8") as fin, open(file_paths['output_file'], "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # current total tokens
            current_total = int(record.get("total_dialogue_token_length") or 0)

            # determine needed tokens
            target_total = sampling.get('target_total_tokens')
            needed = max(0, int(target_total) - current_total) if target_total is not None else 0

            if needed <= 0:
                # write through
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                continue

            acc_tokens = 0
            picked: List[Dict[str, Any]] = []
            seen = set()

            while acc_tokens < needed:
                item = draw_one()
                if item is None:
                    break
                if bool(validation.get('deduplicate_samples', True)):
                    key = hash(item["messages"][0]["content"]) ^ hash(item["messages"][1]["content"]) ^ item["token_length"]
                    if key in seen:
                        continue
                    seen.add(key)
                picked.append(item)
                acc_tokens += int(item.get("token_length", 0))

            if not picked:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                continue

            # split into internal vs new sessions using config ratio
            # ratio=0.0 means all internal insertion, ratio=1.0 means all new sessions
            n = len(picked)
            n_new_sessions = int(round(n * ratio))  
            n_internal = n - n_new_sessions
            internal_items = picked[:n_internal]
            new_items = picked[n_internal:]

            # internal insertion: distribute across sessions (append as new turn)
            sessions = record.get("sessions", [])
            if sessions and internal_items:
                # choose sessions uniformly
                for it in internal_items:
                    sess = rng.choice(sessions)
                    qa_msgs = it["messages"]
                    # ensure exact two messages and roles user->assistant
                    if not (isinstance(qa_msgs, list) and len(qa_msgs) >= 2):
                        continue
                    qa_pair = [
                        {"role": qa_msgs[0].get("role", "user"), "content": qa_msgs[0].get("content", "")},
                        {"role": qa_msgs[1].get("role", "assistant"), "content": qa_msgs[1].get("content", "")},
                    ]
                    insert_internal(sess, qa_pair, int(it.get("token_length", 0)), rng)

            # new sessions: split remaining items into multiple sessions based on average session length
            if new_items:
                # Calculate average session length from existing sessions
                if sessions:
                    avg_length = sum(int(s.get("dialogue_token_length", 0)) for s in sessions) / len(sessions)
                else:
                    # If no existing sessions, use a default average
                    avg_length = 1000  # default fallback
                
                # Group new_items into multiple sessions, each not exceeding avg_length
                new_sessions_data = []  # List of (qa_batch, qa_sum) tuples
                current_batch = []
                current_sum = 0
                
                for it in new_items:
                    qa_msgs = it["messages"]
                    if not (isinstance(qa_msgs, list) and len(qa_msgs) >= 2):
                        continue
                    qa_pair = [
                        {"role": qa_msgs[0].get("role", "user"), "content": qa_msgs[0].get("content", "")},
                        {"role": qa_msgs[1].get("role", "assistant"), "content": qa_msgs[1].get("content", "")},
                    ]
                    item_tokens = int(it.get("token_length", 0))
                    
                    # Check if adding this item would exceed average length
                    if current_sum + item_tokens > avg_length and current_batch:
                        # Save current batch and start a new one
                        new_sessions_data.append((current_batch, current_sum))
                        current_batch = [qa_pair]
                        current_sum = item_tokens
                    else:
                        current_batch.append(qa_pair)
                        current_sum += item_tokens
                
                # Don't forget the last batch
                if current_batch:
                    new_sessions_data.append((current_batch, current_sum))
                
                # Create new sessions and insert them randomly between existing sessions
                if new_sessions_data and len(sessions) >= 2:
                    # Determine valid insertion positions (between sessions, not before first or after last)
                    # Positions: 1, 2, ..., len(sessions)-1 (insert before session at that index)
                    valid_positions = list(range(1, len(sessions)))
                    
                    # Randomly assign insertion positions for each new session
                    for qa_batch, qa_sum in new_sessions_data:
                        # Choose a random position (with replacement, so multiple can go in same gap)
                        insert_pos = rng.choice(valid_positions)
                        
                        new_sess = create_new_session(sessions, qa_batch, qa_sum, 
                                                     insertion.get('new_session_title', 'Pure_QA_Session'), rng)
                        # Add a unique field to identify this as a generated session
                        new_sess["is_generated_qa_session"] = True
                        new_sess["generation_source"] = "stage6_3_supplement"
                        
                        # Insert at the chosen position
                        sessions.insert(insert_pos, new_sess)
                    
                    # Reindex sessions: add session_id field
                    for idx, sess in enumerate(sessions):
                        sess["session_id"] = idx
                    
                    record["sessions"] = sessions
                elif new_sessions_data:
                    # If less than 2 existing sessions, just append (can't insert between)
                    for qa_batch, qa_sum in new_sessions_data:
                        new_sess = create_new_session(sessions, qa_batch, qa_sum,
                                                     insertion.get('new_session_title', 'Pure_QA_Session'), rng)
                        new_sess["is_generated_qa_session"] = True
                        new_sess["generation_source"] = "stage6_3_supplement"
                        sessions.append(new_sess)
                    
                    # Reindex sessions
                    for idx, sess in enumerate(sessions):
                        sess["session_id"] = idx
                    
                    record["sessions"] = sessions

            record["total_dialogue_token_length"] = int(current_total) + int(acc_tokens)

            # update token_cost current stage as zero-cost and accumulate
            token_cost = record.get("token_cost") or {}
            current_stage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "model": None,
                "pricing_available": False,
                "note": "Stage6_3 long: dialogue expansion only",
            }
            if "current_stage" in token_cost:
                token_cost["previous_stage"] = token_cost.get("current_stage")
            token_cost["current_stage"] = current_stage
            record["token_cost"] = token_cost

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Stage6_3 long completed. Wrote {written} records to {file_paths['output_file']}")


if __name__ == "__main__":
    main()


