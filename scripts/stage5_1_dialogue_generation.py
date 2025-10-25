import json
import random
import os
import copy
import traceback
import jsonlines
import logging
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Set, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ConversationConverter

from llm_request import llm_request, calculate_cumulative_cost, _extract_json_from_content
from parallel_utils import parallel_process
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)

logger = logging.getLogger(__name__)

# Load environment variables and configuration
load_dotenv()

RETRY_TIMES = int(os.getenv('RETRY_TIMES'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER'))


# Load configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    full_config = json.load(f)
    config = full_config['STAGE5']['substage1']
    # Get memory_count_range from STAGE4 since STAGE5 uses it for mapping
    stage4_config = full_config['STAGE4']['substage1']
    stage4_memory_count_range = stage4_config['memory_count_range']
    # Get interference_memory_count_range from STAGE4 since STAGE5 uses it for interference memory generation
    config['interference_memory_count_range'] = stage4_config['interference_memory_count_range']

# Read prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")
        raise ValueError(f"Cannot load prompt file {prompt_path}: {e}:{traceback.format_exc()}")

# Load stage5_1 prompts
stage5_1_dialogue_generation_prompt_template = load_prompt(config['prompts']['stage5_1_dialogue_generation'])
stage5_1_memory_validation_prompt_template = load_prompt(config['prompts']['stage5_1_memory_validation'])
stage5_1_interference_memory_generation_prompt_template = load_prompt(config['prompts']['stage5_1_interference_memory_generation'])
stage5_1_dialogue_generation_with_interference_prompt_template = load_prompt(config['prompts']['stage5_1_dialogue_generation_with_interference'])


def format_dialogue_generation_prompt(
    persona_info: str,
    event_lists: List[str],
    stored_memory_points: List[str],
    start_time: str,
    end_time: str,
    user_motivation_type: str,
    dialogue_goal: str,
    dialogue_summary: str,
    memory_points: str,
    turn_num: int
) -> str:
    """Format dialogue generation prompt, fill all placeholders"""
    try:
        # Convert lists to string format
        event_lists_str = '\n'.join(event_lists) if event_lists else "No historical events"
        stored_memory_points_str = '\n'.join(stored_memory_points) if stored_memory_points else "No stored memory points"
        
        # Use format method to fill placeholders
        formatted_prompt = stage5_1_dialogue_generation_prompt_template.format(
            persona_info=persona_info,
            event_lists=event_lists_str,
            stored_memory_points=stored_memory_points_str,
            start_time=start_time,
            end_time=end_time,
            user_motivation_type=user_motivation_type,
            dialogue_goal=dialogue_goal,
            dialogue_summary=dialogue_summary,
            memory_points=memory_points,
            turn_num=turn_num
        )
        
        return formatted_prompt
    except KeyError as e:
        print(f"[ERROR] Prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
        raise ValueError(f"Prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
    except Exception as e:
        print(f"[ERROR] Prompt formatting failed: {e}:{traceback.format_exc()}")
        raise ValueError(f"Prompt formatting failed: {e}:{traceback.format_exc()}")


def format_interference_memory_generation_prompt(
    original_memories: List[Dict],
    persona_info: str,
    existing_memories: List[str],
    interference_memory_count: int
) -> str:
    """Format to generate prompt words that interfere with memory points"""
    try:
        # Build formatted strings for prompt placeholders
        original_memory_lines = [
            f"{idx}. {memory.get('memory_content', '')}"
            for idx, memory in enumerate(original_memories, start=1)
        ]
        memory_type_lines = [
            f"{idx}. {memory.get('memory_type', 'Persona Memory')}"
            for idx, memory in enumerate(original_memories, start=1)
        ]

        original_memory_str = '\n'.join(original_memory_lines) if original_memory_lines else "No original memory points"
        memory_type_str = '\n'.join(memory_type_lines) if memory_type_lines else "No memory types"
        existing_memories_str = '\n'.join(existing_memories) if existing_memories else "No existing memory points"
        
        # Use format method to fill placeholders
        formatted_prompt = stage5_1_interference_memory_generation_prompt_template.format(
            original_memory=original_memory_str,
            memory_type=memory_type_str,
            persona_info=persona_info,
            existing_memories=existing_memories_str,
            interference_memory_count=interference_memory_count
        )
        
        return formatted_prompt
    except KeyError as e:
        print(f"[ERROR] Interference memory generation prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
        raise ValueError(f"Interference memory generation prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
    except Exception as e:
        print(f"[ERROR] Interference memory generation prompt formatting failed: {e}:{traceback.format_exc()}")
        raise ValueError(f"Interference memory generation prompt formatting failed: {e}:{traceback.format_exc()}")


def format_dialogue_generation_with_interference_prompt(
    persona_info: str,
    event_lists: List[str],
    stored_memory_points: List[str],
    start_time: str,
    end_time: str,
    user_motivation_type: str,
    dialogue_goal: str,
    dialogue_summary: str,
    memory_points: str,
    interference_memories: str,
    turn_num: int,
    enforced_assistant_prompts: str,
    enforced_user_constraints: str
) -> str:
    """Format dialogue with interference memory points to generate prompt words"""
    try:
        # Convert lists to string format
        event_lists_str = '\n'.join(event_lists) if event_lists else "No historical events"
        stored_memory_points_str = '\n'.join(stored_memory_points) if stored_memory_points else "No stored memory points"
        
        # Use format method to fill placeholders
        formatted_prompt = stage5_1_dialogue_generation_with_interference_prompt_template.format(
            persona_info=persona_info,
            event_lists=event_lists_str,
            stored_memory_points=stored_memory_points_str,
            start_time=start_time,
            end_time=end_time,
            user_motivation_type=user_motivation_type,
            dialogue_goal=dialogue_goal,
            dialogue_summary=dialogue_summary,
            memory_points=memory_points,
            interference_memories=interference_memories,
            turn_num=turn_num,
            enforced_assistant_prompts=enforced_assistant_prompts,
            enforced_user_constraints=enforced_user_constraints
        )
        
        return formatted_prompt
    except KeyError as e:
        print(f"[ERROR] Dialogue generation with interference prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
        raise ValueError(f"Dialogue generation with interference prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
    except Exception as e:
        print(f"[ERROR] Dialogue generation with interference prompt formatting failed: {e}:{traceback.format_exc()}")
        raise ValueError(f"Dialogue generation with interference prompt formatting failed: {e}:{traceback.format_exc()}")


def format_memory_validation_prompt(
    dialogue_data: Dict,
    memory_points_str: str
) -> str:
    """Format memory validation prompt, fill all placeholders"""
    try:
        # Convert dialogue data to string format
        dialogue_str = json.dumps(dialogue_data, ensure_ascii=False, indent=2)
        
        # Use format method to fill placeholders
        formatted_prompt = stage5_1_memory_validation_prompt_template.format(
            dialogue=dialogue_str,
            memory_points=memory_points_str
        )
        
        return formatted_prompt
    except KeyError as e:
        print(f"[ERROR] Memory validation prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
        raise ValueError(f"Memory validation prompt formatting failed, missing placeholder: {e}:{traceback.format_exc()}")
    except Exception as e:
        print(f"[ERROR] Memory validation prompt formatting failed: {e}:{traceback.format_exc()}")
        raise ValueError(f"Memory validation prompt formatting failed: {e}:{traceback.format_exc()}")


def generate_dialogue_for_initial_event(
    initial_data: Dict,
    event_info: Dict,
    event_type: str
) -> Dict:
    """Generate dialogue for initial event using ConversationConverter from utils.py"""
    print(f"[DEBUG] Generating dialogue for initial event: {event_type}")
    
    # Create ConversationConverter instance, specify correct template directory
    import os
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
    converter = ConversationConverter(prompts_dir=prompts_dir)
    
    # Validate initial_data structure
    if not initial_data:
        print(f"[ERROR] initial_data is empty, cannot generate dialogue")
        raise ValueError(f"initial_data is empty, cannot generate dialogue")
    
    # Generate dialogue based on event type
    if event_type == "initial_fixed":
        # Generate fixed information dialogue
        # initial_data should contain data structure with fixed field
        if not isinstance(initial_data, dict):
            print(f"[ERROR] initial_data is not a dictionary type: {type(initial_data)}")
            raise ValueError(f"initial_data is not a dictionary type: {type(initial_data)}")
        
        dialogue_messages = converter._format_fixed_conversation(initial_data)
    elif event_type == "initial_dynamic":
        # Generate dynamic information dialogue - now initial_data contains all dynamic types
        if not isinstance(initial_data, dict):
            print(f"[ERROR] initial_data is not a dictionary type: {type(initial_data)}")
            raise ValueError(f"initial_data is not a dictionary type: {type(initial_data)}")
        
        dialogue_messages = converter._format_dynamic_conversation(initial_data)
    elif event_type == "initial_preference":
        # Generate preference information dialogue - now initial_data contains all preference types, merge into one dialogue
        if not isinstance(initial_data, dict):
            print(f"[ERROR] initial_data is not a dictionary type: {type(initial_data)}")
            raise ValueError(f"initial_data is not a dictionary type: {type(initial_data)}")
        
        all_preference_conversations = converter._format_preferences_conversation(initial_data)
        if not all_preference_conversations:
            dialogue_messages = []
        else:
            # Merge all preference dialogues into one dialogue
            dialogue_messages = []
            for preference_conversation in all_preference_conversations:
                dialogue_messages.extend(preference_conversation)
    else:
        print(f"[ERROR] Unknown initial event type: {event_type}")
        raise ValueError(f"Unknown initial event type: {event_type}")
    
    # Check if dialogue messages are empty
    if not dialogue_messages:
        print("[ERROR] Generated dialogue messages are empty, this should not happen in production")
        raise ValueError("Generated dialogue messages are empty - this indicates a serious generation failure")
    
    # Convert dialogue messages to stage5_1 required format
    dialogue_result = {}
    
    # Check if dialogue messages are empty
    if not dialogue_messages:
        print("[ERROR] Dialogue messages are empty, this should not happen in production")
        raise ValueError("Dialogue messages are empty - this indicates a serious generation failure")
    
    # Calculate dialogue turns - handle possible incomplete turns
    # Each dialogue turn is usually user-assistant, but there may be additional user messages
    turn_num = len(dialogue_messages) // 2
    
    for i in range(turn_num):
        turn_key = f"dialogue_turn_{i + 1}"
        user_message = dialogue_messages[i * 2]
        assistant_message = dialogue_messages[i * 2 + 1]
        
        dialogue_result[turn_key] = [
            {"role": "user", "content": user_message["content"]},
            {"role": "assistant", "content": assistant_message["content"]}
        ]
    
    # If there are additional user messages (like the last one), add to the last turn
    if len(dialogue_messages) > turn_num * 2:
        last_user_message = dialogue_messages[-1]
        if last_user_message["role"] == "user":
            # Add additional user message to the last turn
            last_turn_key = f"dialogue_turn_{turn_num}"
            if last_turn_key in dialogue_result:
                dialogue_result[last_turn_key].append({
                    "role": "user", 
                    "content": last_user_message["content"]
                })
    
    # If no dialogue is generated, report error and exit
    if not dialogue_result:
        print("[ERROR] utils.py failed to generate dialogue, cannot continue processing")
        raise ValueError("utils.py failed to generate dialogue, cannot continue processing")
    
    print(f"[DEBUG] Initial event dialogue generation completed, total {len(dialogue_result)} dialogue turns")
    return dialogue_result


def map_range(x, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    return round(out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min))


def assign_dialogue_timestamps(
    dialogue_data: Dict, 
    start_time_str: str, 
    end_time_str: str, 
    min_gap_seconds: int = 60
) -> Dict:
    """Assign timestamps for multi-turn dialogue data"""
    print("[DEBUG] Assigning timestamps for dialogue...")
    
    time_format = "%b %d, %Y, %H:%M:%S"
    try:
        start_time = datetime.strptime(start_time_str, time_format)
        end_time = datetime.strptime(end_time_str, time_format)
    except ValueError as e:
        print(f"[DEBUG] Time format error: {e}:{traceback.format_exc()}")
        return dialogue_data

    total_duration_seconds = (end_time - start_time).total_seconds()
    if total_duration_seconds < 0:
        print("[DEBUG] End time cannot be earlier than start time")
        return dialogue_data

    # Sort dialogue turns by numeric suffix
    dialogue_turns = sorted(dialogue_data.keys(), key=lambda k: int(k.split('_')[-1]))
    num_turns = len(dialogue_turns)

    if num_turns == 0:
        return dialogue_data

    if num_turns == 1:
        dialogue_data[dialogue_turns[0]].append({'timestamp': start_time.strftime(time_format)})
        return dialogue_data

    num_intervals = num_turns - 1
    total_min_gap_time = num_intervals * min_gap_seconds
    slack_time = total_duration_seconds - total_min_gap_time

    if slack_time < 0:
        print(f"[DEBUG] Total duration insufficient to support {num_turns} dialogue turns")
        return dialogue_data

    # Use Dirichlet distribution to allocate remaining time
    random_additions = np.random.dirichlet(np.ones(num_intervals)) * slack_time
    interval_durations = [min_gap_seconds + addition for addition in random_additions]

    updated_dialogue_data = {}
    current_time = start_time

    # Assign timestamps for each dialogue turn
    for i, turn_key in enumerate(dialogue_turns):
        updated_dialogue_data[turn_key] = dialogue_data[turn_key] + [
            {'timestamp': current_time.strftime(time_format)}
        ]
        if i < num_intervals:
            current_time += timedelta(seconds=interval_durations[i])

    # Ensure the last timestamp does not exceed end time
    if current_time != end_time:
        updated_dialogue_data[dialogue_turns[-1]][-1]['timestamp'] = end_time.strftime(time_format)

    return updated_dialogue_data


def assign_memory_point_timestamps(
    memory_points_data: List[Dict],
    dialogue_data: Dict,
    dialogue_turn_num: int
) -> List[Dict]:
    """Assign corresponding timestamps for memory points"""
    print("[DEBUG] Assigning timestamps for memory points...")
    
    new_memory_points_data = []
    cnt = 1
    
    # Get dialogue end time
    dialogue_turns = sorted(dialogue_data.keys(), key=lambda k: int(k.split('_')[-1]))
    end_timestamp = None
    if dialogue_turns:
        last_turn = dialogue_turns[-1]
        if dialogue_data[last_turn] and len(dialogue_data[last_turn]) > 0:
            end_timestamp = dialogue_data[last_turn][-1].get("timestamp")
    
    for item in memory_points_data:
        # Check if memory_source field exists
        if not item.get('memory_source'):
            # If no memory_source, skip this memory point
            continue
        
        # For all memory point types, timestamps need to be assigned
        if item['memory_source'] in ['system', 'secondary', 'interference']:
            # 使用对话结束时间而不是默认时间戳
            if 'timestamp' not in item or item['timestamp'] == "Jan 15, 2025, 14:30:00":
                item['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
            item["index"] = cnt
            cnt += 1
            new_memory_points_data.append(item)
            continue
            
        # For other cases, if dialogue_turn information exists, assign timestamp
        if 'dialogue_turn' in item:
            dialogue_turn = item['dialogue_turn']
            if not (0 < dialogue_turn <= dialogue_turn_num):
                print(f"[DEBUG] Dialogue turn out of range: {dialogue_turn}")
                continue
                
            dialogue_turn_key = f"dialogue_turn_{dialogue_turn}"
            if dialogue_turn_key not in dialogue_data:
                print(f"[DEBUG] Dialogue turn does not exist: {dialogue_turn_key}")
                continue
                
            time_point = dialogue_data[dialogue_turn_key][-1]["timestamp"]
            item["timestamp"] = time_point
            item["index"] = cnt
            cnt += 1
            new_memory_points_data.append(item)
        else:
            # No dialogue_turn information, use end timestamp
            if 'timestamp' not in item or item['timestamp'] == "Jan 15, 2025, 14:30:00":
                item['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
            item["index"] = cnt
            cnt += 1
            new_memory_points_data.append(item)
        
    return new_memory_points_data


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_dialogue_for_event(
    persona_info: str,
    current_events: List[str],
    stored_memory_points: List[str],
    event_info: Dict,
    dialogue_turn_num: int = None,
    persona_data: Dict = None,
    previous_cost: Dict = None
) -> Dict:
    """Generate dialogue for a single event"""
    print(f"[DEBUG] Generating dialogue for event: {event_info.get('event_name', 'unknown')}")
    
    # Check if it's an initial event - refer to stage4's recognition method
    is_init_event = False
    initial_data = None
    event_type = ""
    
    if 'initial_fixed' in event_info:
        is_init_event = True
        initial_data = event_info['initial_fixed']
        event_type = "initial_fixed"
    elif 'initial_dynamic' in event_info:
        is_init_event = True
        initial_data = event_info['initial_dynamic']
        event_type = "initial_dynamic"
    elif 'initial_preference' in event_info:
        is_init_event = True
        initial_data = event_info['initial_preference']
        event_type = "initial_preference"
    
    if is_init_event:
        print(f"[DEBUG] Detected initial event: {event_info.get('event_name', 'unknown')} - {event_type}")
        
        if initial_data is None:
            print("[ERROR] Initial event data not found, cannot generate dialogue")
            raise ValueError("Initial event data not found, cannot generate dialogue")
        
        # Use utils.py to generate initial event dialogue
        try:
            dialogue_result = generate_dialogue_for_initial_event(initial_data, event_info, event_type)
            return {
                "dialogue": dialogue_result,
                "interference_memories": []
            }
        except Exception as e:
            print(f"[ERROR] utils.py failed to generate initial event dialogue: {e}:{traceback.format_exc()}")
            raise ValueError(f"utils.py failed to generate initial event dialogue: {e}:{traceback.format_exc()}")
    
    # Non-initial event, use original AI generation method
    if not os.getenv('OPENAI_API_KEY'):
        print("[ERROR] API key not set, this should not happen in production")
        raise ValueError("OPENAI_API_KEY environment variable is not set - this is required for dialogue generation")
    
    # Build memory points string
    memory_points_str = '\n'.join(
        [json.dumps(i, ensure_ascii=False) for i in event_info['dialogue_info']['memory_points']]
    )
    
    # Generate interference memory points (init events do not use interference memory points)
    interference_memories: List[Dict] = []
    interference_memories_str = ""
    enforced_assistant_prompts = "无需额外干扰指令。保持对真实记忆的正常引用。"
    enforced_user_constraints = "用户发言可自然引用真实记忆点，无需特殊限制。"
    
    # Check if it's an init event, if so, do not use interference memory points
    # Use the same recognition method as above
    is_init_event = ('initial_fixed' in event_info or 'initial_dynamic' in event_info or 'initial_preference' in event_info)
    
    if not is_init_event and event_info['dialogue_info']['memory_points']:
        available_memories = event_info['dialogue_info']['memory_points']
        max_possible = min(len(available_memories), config['interference_memory_count_range'][1])

        if max_possible > 0:
            min_possible = min(config['interference_memory_count_range'][0], max_possible)
            if min_possible < 1:
                min_possible = 1
            interference_count = random.randint(min_possible, max_possible)

            selected_indices = random.sample(range(len(available_memories)), interference_count)
            selected_memories = [available_memories[idx] for idx in selected_indices]
            selected_lookup = {i + 1: selected_memories[i] for i in range(len(selected_memories))}

            print("[DEBUG] Generating interference memory points for selected memories:")
            for idx, mem in enumerate(selected_memories, start=1):
                print(f"        [DEBUG] Target #{idx}: {mem.get('memory_content', 'unknown')}")

            interference_memories, interference_cost = generate_interference_memories(
                target_memories=selected_memories,
                persona_info=persona_info,
                existing_memories=available_memories,
                interference_count=interference_count,
                previous_cost=previous_cost
            )

            print(f"[DEBUG] Generated {len(interference_memories)} interference memory points")

            valid_interference_memories = []
            used_sources = set()

            for memory in interference_memories:
                source_idx = memory.get('source_memory_index')
                if not isinstance(source_idx, int):
                    print(f"[WARN] Interference memory missing valid source index: {memory}")
                    continue
                referenced_memory = selected_lookup.get(source_idx)
                if not referenced_memory:
                    print(f"[WARN] Source index {source_idx} not found in selected memories")
                    continue
                if source_idx in used_sources:
                    print(f"[WARN] Duplicate interference for source index {source_idx}, skipping extra entry")
                    continue

                memory['memory_type'] = referenced_memory.get('memory_type', memory.get('memory_type', 'Persona Memory'))
                memory['is_update'] = memory.get('is_update', 'False')
                memory['original_memories'] = memory.get('original_memories', [])
                memory['memory_source'] = 'interference'
                memory['reference_memory_content'] = referenced_memory.get('memory_content', '')
                memory['reference_memory_index'] = referenced_memory.get('index')
                memory['reference_memory_type'] = referenced_memory.get('memory_type', '')

                valid_interference_memories.append(memory)
                used_sources.add(source_idx)

            interference_memories = valid_interference_memories

            # Re-index for downstream consumers
            for i, memory in enumerate(interference_memories):
                memory['index'] = i + 1

            if interference_memories:
                interference_details = []
                assistant_prompt_lines = [
                    "- 在对话中以十分自信的语气提及并强化以下干扰记忆点，每条至少深入展开一次并与真实记忆形成对比："
                ]
                user_constraint_lines = [
                    "- 用户发言必须忽略并不回应以下干扰记忆点，即使助手主动提及也不要确认或纠正："
                ]

                for idx, memory in enumerate(interference_memories, start=1):
                    source_idx = memory.get('source_memory_index')
                    referenced_memory = selected_lookup.get(source_idx, {})
                    referenced_content = referenced_memory.get('memory_content', 'unknown')
                    conflict_angle = memory.get('conflict_angle', '未说明冲突角度')

                    interference_details.append(
                        f"- 干扰记忆#{idx}（针对真实记忆#{source_idx}: {referenced_content}，冲突角度：{conflict_angle}）：{memory.get('memory_content', '')}"
                    )
                    assistant_prompt_lines.append(
                        f"  {idx}. 将干扰记忆“{memory.get('memory_content', '')}”作为事实描述，并与真实记忆“{referenced_content}”对照，突出冲突角度：{conflict_angle}。"
                    )
                    user_constraint_lines.append(
                        f"  {idx}. 忽略助手提到的“{memory.get('memory_content', '')}”，保持原本的真实记忆叙述。"
                    )
                
                assistant_prompt_lines.extend([
                    "- 干扰记忆必须贯穿多轮回复，可在总结阶段再次重申干扰信息。",
                    "- 禁止在对话中质疑干扰记忆的真实性或承认其为猜测。"
                ])

                user_constraint_lines.extend([
                    "- 用户可以继续引用以下真实记忆点，保持自然交流："
                ])

                for idx, memory in enumerate(available_memories, start=1):
                    user_constraint_lines.append(f"  真记忆#{idx}: {memory.get('memory_content', '')}")

                interference_memories_str = '\n'.join(interference_details)
                enforced_assistant_prompts = '\n'.join(assistant_prompt_lines)
                enforced_user_constraints = '\n'.join(user_constraint_lines)

            else:
                print("[WARN] No valid interference memories generated after filtering")
    
    # Always use the version with interference memory points (interference_memories_str may be empty)
    system_prompt = format_dialogue_generation_with_interference_prompt(
        persona_info=persona_info,
        event_lists=current_events,
        stored_memory_points=stored_memory_points,
        start_time=event_info["dialogue_info"]['start_time_point'],
        end_time=event_info["dialogue_info"]['end_time_point'],
        user_motivation_type=event_info["dialogue_info"]['user_motivation_type'],
        dialogue_goal=event_info["dialogue_info"].get('dialogue_goal', ''),
        dialogue_summary=event_info["dialogue_info"]['dialogue_summary'],
        memory_points=memory_points_str,
        interference_memories=interference_memories_str,
        turn_num=dialogue_turn_num,
        enforced_assistant_prompts=enforced_assistant_prompts,
        enforced_user_constraints=enforced_user_constraints
    )
    
    # Build simplified user input content (as main information is already in system prompt)
    user_content = "Please generate dialogue content that meets the requirements based on the above information."
    
    try:
        print("[DEBUG] Sending dialogue generation request to LLM...")

        # Find JSON part in thought chain output
        markers = ["Dialogue Generation Result", "Generated Dialogue", "JSON Result", "Final JSON", "Full JSON", "Generation Result"]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=markers)
        
        if 'interference_cost' in locals():

            combined_cost = {
                "input_tokens": cost_info.get("input_tokens", 0) + interference_cost.get("current_stage", {}).get("input_tokens", 0),
                "output_tokens": cost_info.get("output_tokens", 0) + interference_cost.get("current_stage", {}).get("output_tokens", 0),
                "total_tokens": cost_info.get("total_tokens", 0) + interference_cost.get("current_stage", {}).get("total_tokens", 0),
                "total_cost_usd": cost_info.get("total_cost_usd", 0) + interference_cost.get("current_stage", {}).get("total_cost_usd", 0),
                "pricing_available": cost_info.get("pricing_available", False)
            }
            token_cost = calculate_cumulative_cost(previous_cost, combined_cost)
        else:
            token_cost = calculate_cumulative_cost(previous_cost, cost_info)
        
        current_cost = cost_info
        cumulative_cost = token_cost.get('cumulative', {})
        print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
              f"Output: {current_cost.get('output_tokens', 'N/A')}, "
              f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
        print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
              f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        content = raw_response
        print(f"[DEBUG] API response content: {content[:100]}...")
        
        # Try to parse JSON
        json_content = content
        
        # Find JSON part in thought chain output
        # markers = ["Dialogue Generation Result", "Generated Dialogue", "JSON Result", "Final JSON", "Full JSON", "Generation Result"]
        for marker in markers:
            if marker in json_content:
                parts = json_content.split(marker, 1)
                if len(parts) > 1:
                    json_content = parts[1].strip()
                    break
        
        # Handle possible markdown code blocks
        if '```json' in json_content:
            start_idx = json_content.find('```json') + 7
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        elif '```' in json_content:
            start_idx = json_content.find('```') + 3
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        # Try to find start and end of JSON object
        if '{' in json_content and '}' in json_content:
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}') + 1
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        try:
            dialogue_result = json.loads(json_content)
            print(f"[DEBUG] Successfully parsed dialogue JSON")
            
            # Validate dialogue format - skip validation for initial events
            is_init_event = ('initial_fixed' in event_info or 'initial_dynamic' in event_info or 'initial_preference' in event_info)
            
            if dialogue_turn_num is not None and not is_init_event:  # Only validate for non-initial events
                if len(dialogue_result.keys()) != dialogue_turn_num:
                    print(f"[ERROR] Dialogue turn count mismatch, expected {dialogue_turn_num}, actual {len(dialogue_result.keys())}")
                    raise ValueError(f"Dialogue turn count mismatch - expected {dialogue_turn_num} turns but got {len(dialogue_result.keys())} turns")
                
                expected_keys = set([f"dialogue_turn_{i+1}" for i in range(dialogue_turn_num)])
                if dialogue_result.keys() != expected_keys:
                    print(f"[ERROR] Dialogue turn keys mismatch, expected {expected_keys}, got {set(dialogue_result.keys())}")
                    raise ValueError(f"Dialogue turn keys mismatch - expected {expected_keys} but got {set(dialogue_result.keys())}")
            else:
                if is_init_event:
                    print(f"[DEBUG] Initial event, skipping dialogue turn count validation, actual turns: {len(dialogue_result.keys())}")
                else:
                    print(f"[DEBUG] No dialogue turn count specified, skipping validation, actual turns: {len(dialogue_result.keys())}")
            
            # For non-initial events, perform memory validation and include its cost
            if not is_init_event:
                print("[DEBUG] Performing memory validation for non-initial event...")
                original_memory_points = event_info['dialogue_info']['memory_points']
                memory_points_str = '\n' + '\n'.join(
                    [json.dumps(i, ensure_ascii=False) for i in original_memory_points]
                )
                
                validated_memory_points, validation_cost = validate_memory_points(
                    dialogue_result, memory_points_str, original_memory_points, previous_cost
                )
                
                # Merge validation cost into total cost
                if validation_cost:
                    current_cost = validation_cost.get("current_stage", {})
                    combined_cost = {
                        "input_tokens": token_cost.get("current_stage", {}).get("input_tokens", 0) + current_cost.get("input_tokens", 0),
                        "output_tokens": token_cost.get("current_stage", {}).get("output_tokens", 0) + current_cost.get("output_tokens", 0),
                        "total_tokens": token_cost.get("current_stage", {}).get("total_tokens", 0) + current_cost.get("total_tokens", 0),
                        "total_cost_usd": token_cost.get("current_stage", {}).get("total_cost_usd", 0) + current_cost.get("total_cost_usd", 0),
                        "pricing_available": token_cost.get("current_stage", {}).get("pricing_available", False)
                    }
                    token_cost = calculate_cumulative_cost(previous_cost, combined_cost)
                    print(f"[DEBUG] Memory validation cost merged - Input: {current_cost.get('input_tokens', 0)}, "
                          f"Output: {current_cost.get('output_tokens', 0)}, "
                          f"Cost: ${current_cost.get('total_cost_usd', 0)}")
            
            # Return dialogue result and interference memory point information
            result = {
                "dialogue": dialogue_result,
                "interference_memories": interference_memories,
                "token_cost": token_cost
            }
            
            # For non-initial events, also return validated memory points
            if not is_init_event:
                result["validated_memory_points"] = validated_memory_points
            
            return result
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}:{traceback.format_exc()}, trying to extract with regex")
            # Try to extract JSON with a more strict method
            import re
            json_pattern = r'({[\s\S]*})'
            match = re.search(json_pattern, json_content)
            if match:
                try:
                    potential_json = match.group(1)
                    dialogue_result = json.loads(potential_json)
                    print(f"[DEBUG] Successfully extracted dialogue JSON using regex")
                    return {
                        "dialogue": dialogue_result,
                        "interference_memories": interference_memories,
                        "token_cost": token_cost
                    }
                except json.JSONDecodeError:
                    print(f"[DEBUG] Content extracted by regex is not a valid JSON")
            
            # If all attempts fail, raise error instead of returning default dialogue
            print(f"[ERROR] All JSON parsing attempts failed")
            raise ValueError("All JSON parsing attempts failed - this indicates a serious generation or parsing issue")
        
    except Exception as e:
        print(f"[ERROR] Dialogue generation failed: {e}:{traceback.format_exc()}")
        
        empty_cost = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "total_cost_usd": 0, "pricing_available": False
        }
        token_cost = calculate_cumulative_cost(previous_cost, empty_cost)
        
        raise ValueError(f"Dialogue generation failed: {e}:{traceback.format_exc()}")


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_interference_memories(
    target_memories: List[Dict],
    persona_info: str,
    existing_memories: List[Dict],
    interference_count: int,
    previous_cost: Dict = None
) -> Tuple[List[Dict], Dict]:
    """Generate interference memory points based on multiple original memories"""
    target_descriptions = ', '.join([m.get('memory_content', 'unknown') for m in target_memories])
    print(f"[DEBUG] Generating interference memory points for memories: {target_descriptions}")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("[DEBUG] API key not set, returning empty interference memory points")
        empty_cost = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "total_cost_usd": 0, "pricing_available": False
        }
        token_cost = calculate_cumulative_cost(previous_cost, empty_cost)
        return [], token_cost
    
    # Convert existing memory points to string list
    existing_memories_str = [f"- {memory.get('memory_content', '')}" for memory in existing_memories]
    
    # Format interference memory generation prompt
    system_prompt = format_interference_memory_generation_prompt(
        original_memories=target_memories,
        persona_info=persona_info,
        existing_memories=existing_memories_str,
        interference_memory_count=interference_count
    )
    
    # Build user input content
    user_content = "Please generate interference memory points based on the above information."
    
    try:
        print("[DEBUG] Sending interference memory generation request to LLM...")

        # Find JSON part in thought chain output
        markers = ["Interference Memory Generation Result", "Generated Interference Memory", "JSON Result", "Final JSON", "Full JSON", "Generation Result"]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=markers)
        
        token_cost = calculate_cumulative_cost(previous_cost, cost_info)
        
        content = raw_response
        print(f"[DEBUG] API response content: {content[:100]}...")
        
        # Try to parse JSON
        json_content = content
        
        # Find JSON part in thought chain output
        # markers = ["Interference Memory Generation Result", "Generated Interference Memory", "JSON Result", "Final JSON", "Full JSON", "Generation Result"]
        for marker in markers:
            if marker in json_content:
                parts = json_content.split(marker, 1)
                if len(parts) > 1:
                    json_content = parts[1].strip()
                    break
        
        # Handle possible markdown code blocks
        if '```json' in json_content:
            start_idx = json_content.find('```json') + 7
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        elif '```' in json_content:
            start_idx = json_content.find('```') + 3
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        # Try to find start and end of JSON object
        if '{' in json_content and '}' in json_content:
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}') + 1
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        try:
            generation_result = json.loads(json_content)
            interference_memories = generation_result.get('interference_memories', [])
            print(f"[DEBUG] Successfully parsed interference memory generation JSON, generated {len(interference_memories)} interference memory points")

            # Ensure interference memory points have correct format
            for memory in interference_memories:
                if 'memory_type' not in memory:
                    source_index = memory.get('source_memory_index')
                    referenced_memory = None
                    if isinstance(source_index, int) and 1 <= source_index <= len(target_memories):
                        referenced_memory = target_memories[source_index - 1]
                    if referenced_memory:
                        memory['memory_type'] = referenced_memory.get('memory_type', 'Persona Memory')
                    else:
                        memory['memory_type'] = target_memories[0].get('memory_type', 'Persona Memory') if target_memories else 'Persona Memory'
                if 'is_update' not in memory:
                    memory['is_update'] = 'False'
                if 'original_memories' not in memory:
                    memory['original_memories'] = []
                memory['memory_source'] = 'interference'

            return interference_memories, token_cost
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}:{traceback.format_exc()}, trying to extract with regex")
            # Try to extract JSON with a more strict method
            import re
            json_pattern = r'({[\s\S]*})'
            match = re.search(json_pattern, json_content)
            if match:
                try:
                    potential_json = match.group(1)
                    generation_result = json.loads(potential_json)
                    interference_memories = generation_result.get('interference_memories', [])
                    print(f"[DEBUG] Successfully extracted interference memory generation JSON using regex")

                    # Ensure interference memory points have correct format
                    for memory in interference_memories:
                        if 'memory_type' not in memory:
                            source_index = memory.get('source_memory_index')
                            referenced_memory = None
                            if isinstance(source_index, int) and 1 <= source_index <= len(target_memories):
                                referenced_memory = target_memories[source_index - 1]
                            if referenced_memory:
                                memory['memory_type'] = referenced_memory.get('memory_type', 'Persona Memory')
                            else:
                                memory['memory_type'] = target_memories[0].get('memory_type', 'Persona Memory') if target_memories else 'Persona Memory'
                        if 'is_update' not in memory:
                            memory['is_update'] = 'False'
                        if 'original_memories' not in memory:
                            memory['original_memories'] = []
                        memory['memory_source'] = 'interference'

                    return interference_memories, token_cost
                except json.JSONDecodeError:
                    print(f"[DEBUG] Content extracted by regex is not a valid JSON")
            
            # If all attempts fail, return empty list
                    print(f"[ERROR] All JSON parsing attempts failed, returning empty interference memory point list")
        raise
        
    except Exception as e:
        print(f"[ERROR] Interference memory generation failed: {e}:{traceback.format_exc()}")
        raise


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def validate_memory_points(
    dialogue_data: Dict,
    memory_points_str: str,
    original_memory_points: List[Dict] = None,
    previous_cost: Dict = None
) -> Tuple[List[Dict], Dict]:
    """Validate and correct memory points"""
    print("[DEBUG] Validating and correcting memory points...")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("[DEBUG] API key not set, returning original memory points")
        empty_cost = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "total_cost_usd": 0, "pricing_available": False
        }
        token_cost = calculate_cumulative_cost(previous_cost, empty_cost)
        return [], token_cost
    
    # Format memory validation prompt
    system_prompt = format_memory_validation_prompt(
        dialogue_data=dialogue_data,
        memory_points_str=memory_points_str
    )
    
    # Build simplified user input content (as main information is already in system prompt)
    user_content = "Please validate and correct memory points based on the above information."
    
    try:
        print("[DEBUG] Sending memory validation request to LLM...")

        # Find JSON part in thought chain output
        markers = ["Memory Validation Result", "Corrected Memory Points", "JSON Result", "Final JSON", "Full JSON", "Validation Result"]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=markers)
        
        token_cost = calculate_cumulative_cost(previous_cost, cost_info)
        
        content = raw_response
        print(f"[DEBUG] API response content: {content[:100]}...")
        
        # Try to parse JSON
        json_content = content
        
        for marker in markers:
            if marker in json_content:
                parts = json_content.split(marker, 1)
                if len(parts) > 1:
                    json_content = parts[1].strip()
                    break
        
        # Handle possible markdown code blocks
        if '```json' in json_content:
            start_idx = json_content.find('```json') + 7
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        elif '```' in json_content:
            start_idx = json_content.find('```') + 3
            end_idx = json_content.rfind('```')
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        # Try to find start and end of JSON object
        if '{' in json_content and '}' in json_content:
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}') + 1
            if end_idx > start_idx:
                json_content = json_content[start_idx:end_idx].strip()
        
        try:
            validation_result = json.loads(json_content)
            memory_points = validation_result.get('memory_points', [])
            print(f"[DEBUG] Successfully parsed memory validation JSON")
            
            # If original memory points exist, maintain original classification annotations
            if original_memory_points:
                # Create a map of original memory points to maintain classification
                original_memory_map = {}
                for mp in original_memory_points:
                    content = mp.get('memory_content', '')
                    original_memory_map[content] = mp
                
                # Maintain original classification annotations for validated memory points
                for mp in memory_points:
                    content = mp.get('memory_content', '')
                    if content in original_memory_map:
                        original_mp = original_memory_map[content]
                        # Maintain original classification annotations
                        mp['memory_source'] = original_mp.get('memory_source', 'secondary')
                        mp['is_update'] = original_mp.get('is_update', 'False')
                        mp['original_memories'] = original_mp.get('original_memories', [])
                    else:
                        # New memory points default to secondary memory points
                        mp['memory_source'] = 'secondary'
            
            # Ensure all memory points have correct format
            for mp in memory_points:
                # Ensure other necessary fields exist
                if 'memory_type' not in mp:
                    mp['memory_type'] = 'Persona Memory'
                if 'is_update' not in mp:
                    mp['is_update'] = 'False'
                if 'original_memories' not in mp:
                    mp['original_memories'] = []
            
            return memory_points, token_cost
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}:{traceback.format_exc()}, trying to extract with regex")
            # Try to extract JSON with a more strict method
            import re
            json_pattern = r'({[\s\S]*})'
            match = re.search(json_pattern, json_content)
            if match:
                try:
                    potential_json = match.group(1)
                    validation_result = json.loads(potential_json)
                    memory_points = validation_result.get('memory_points', [])
                    print(f"[DEBUG] Successfully extracted memory validation JSON using regex")
                    
                    # Ensure all memory points have correct format
                    # Get dialogue end time
                    dialogue_turns = sorted(dialogue_data.keys(), key=lambda k: int(k.split('_')[-1]))
                    end_timestamp = None
                    if dialogue_turns:
                        last_turn = dialogue_turns[-1]
                        if dialogue_data[last_turn] and len(dialogue_data[last_turn]) > 0:
                            end_timestamp = dialogue_data[last_turn][-1].get("timestamp")
                    
                    for mp in memory_points:
                        if 'memory_source' not in mp:
                            mp['memory_source'] = 'secondary'
                        if 'memory_type' not in mp:
                            mp['memory_type'] = 'persona'
                        if 'is_update' not in mp:
                            mp['is_update'] = 'False'
                        if 'original_memories' not in mp:
                            mp['original_memories'] = []
                        if 'timestamp' not in mp or mp['timestamp'] == "Jan 15, 2025, 14:30:00":
                            mp['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
                    
                    return memory_points, token_cost
                except json.JSONDecodeError:
                    print(f"[DEBUG] Content extracted by regex is not a valid JSON")
            
            # If all attempts fail, return empty list
                    print(f"[ERROR] All JSON parsing attempts failed, returning empty memory point list")
        raise
        
    except Exception as e:
        print(f"[ERROR] Memory validation failed: {e}:{traceback.format_exc()}")
        raise


def process_single_user(user_data: Dict, test_mode: bool = False, test_event_count: int = None) -> Dict:
    """Process single user data, generate dialogue"""
    print(f"[DEBUG] Processing user: {user_data.get('uuid', 'unknown')}")
    
    user_id = user_data["uuid"]
    persona_info = user_data["persona_info"]
    events = user_data["event_list"]
    
    previous_cost = user_data.get('token_cost')
    
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }
    
    # Test mode: randomly select a specified number of events
    if test_mode and test_event_count is not None:
        print(f"[DEBUG] Test mode: randomly selecting {test_event_count} out of {len(events)} events")
        if len(events) > test_event_count:
            # Randomly select a specified number of events
            selected_events = random.sample(events, test_event_count)
            # Sort by original order to maintain timeline consistency
            selected_events.sort(key=lambda x: x.get('event_time', ''))
            events = selected_events
            print(f"[DEBUG] Test mode: {len(events)} events selected")
        else:
            print(f"[DEBUG] Test mode: not enough events, using all {len(events)} events")
    
    # Extract persona_data for initial event dialogue generation
    persona_data = user_data.get("profile", {})
    
    new_data = copy.deepcopy(user_data)
    new_data["event_list"] = []
    
    # Initialize memory points and event list
    current_memory_points = []
    current_events = []
    
    # Process each event
    for event in events:
        print(f"[DEBUG] Processing event: {event.get('event_name', 'unknown')}")
        
        event_data = copy.deepcopy(event)
        
        # Process career event
        if event["event_type"] == "career_event":
            event["event_description"] = f"{event['stage_description']} {event['stage_result']}"
        
        # Check if it's an initial event - refer to stage4's recognition method
        is_init_event = ('initial_fixed' in event or 'initial_dynamic' in event or 'initial_preference' in event)
        
        if is_init_event:
            print("[DEBUG] Initial event, using utils generated memory points and dialogue")
            # Determine the initial event type and data
            if 'initial_fixed' in event:
                initial_data = event['initial_fixed']
                event_type = "initial_fixed"
            elif 'initial_dynamic' in event:
                initial_data = event['initial_dynamic']
                event_type = "initial_dynamic"
            elif 'initial_preference' in event:
                initial_data = event['initial_preference']
                event_type = "initial_preference"
            else:
                print("[ERROR] Initial event type not recognized")
                raise ValueError("Initial event type not recognized")
            
            # For initial events, calculate dialogue turns based on actual generated dialogue
            dialogue_result = generate_dialogue_for_initial_event(initial_data, event, event_type)
            dialogue_turn_num = len(dialogue_result.keys())
            print(f"[DEBUG] Actual dialogue turns for initial event: {dialogue_turn_num}")
            
            memory_points = event['dialogue_info']['memory_points']
            
            # Add system memory point annotations for init events
            for mp in memory_points:
                mp['memory_source'] = 'system'
            
            # Assign timestamps for memory points
            memory_points = assign_memory_point_timestamps(
                memory_points_data=memory_points,
                dialogue_data=dialogue_result,
                dialogue_turn_num=dialogue_turn_num
            )
        else:
            # Non-initial event, calculate dialogue turns based on memory count
            memory_count = event["dialogue_info"]['memory_points_count']
            
            # Handle cases where memory_count is outside the expected range
            if memory_count < stage4_memory_count_range[0]:
                # If memory count is too low, use minimum dialogue turns
                dialogue_turn_num = config['dialogue_turn_range'][0]
                print(f"[DEBUG] Memory count {memory_count} below range [{stage4_memory_count_range[0]}, {stage4_memory_count_range[1]}], using minimum dialogue turns: {dialogue_turn_num}")
            elif memory_count > stage4_memory_count_range[1]:
                # If memory count is too high, use maximum dialogue turns
                dialogue_turn_num = config['dialogue_turn_range'][1]
                print(f"[DEBUG] Memory count {memory_count} above range [{stage4_memory_count_range[0]}, {stage4_memory_count_range[1]}], using maximum dialogue turns: {dialogue_turn_num}")
            else:
                # Normal case: use map_range
                dialogue_turn_num = map_range(
                    memory_count,
                    stage4_memory_count_range[0],
                    stage4_memory_count_range[1],
                    config['dialogue_turn_range'][0],
                    config['dialogue_turn_range'][1]
                )
                print(f"[DEBUG] Memory count {memory_count} in range, mapped to dialogue turns: {dialogue_turn_num}")
        
        # Generate dialogue
        dialogue_result = generate_dialogue_for_event(
            persona_info,
            current_events,
            current_memory_points,
            event,
            dialogue_turn_num,
            persona_data,
            previous_cost
        )
        
        # Extract dialogue, interference memory points, validated memory points and cost information
        dialogue = dialogue_result["dialogue"]
        interference_memories = dialogue_result.get("interference_memories", [])
        validated_memory_points = dialogue_result.get("validated_memory_points", None)
        dialogue_cost = dialogue_result.get("token_cost", {})
        
        if dialogue_cost:
            current_cost = dialogue_cost.get("current_stage", {})
            current_stage_total_cost["input_tokens"] += current_cost.get("input_tokens", 0)
            current_stage_total_cost["output_tokens"] += current_cost.get("output_tokens", 0)
            current_stage_total_cost["total_tokens"] += current_cost.get("total_tokens", 0)
            current_stage_total_cost["total_cost_usd"] += current_cost.get("total_cost_usd", 0.0)
            if current_stage_total_cost["model"] is None:
                current_stage_total_cost["model"] = current_cost.get("model")
            current_stage_total_cost["pricing_available"] = current_cost.get("pricing_available", False)
        
        # Assign timestamps for dialogue
        dialogue = assign_dialogue_timestamps(
            dialogue_data=dialogue,
            start_time_str=event["dialogue_info"]['start_time_point'],
            end_time_str=event["dialogue_info"]['end_time_point'],
        )
        
        # Check if it's an initial event - refer to stage4's recognition method
        is_init_event = ('initial_fixed' in event or 'initial_dynamic' in event or 'initial_preference' in event)
        
        if is_init_event:
            print("[DEBUG] Initial event, using utils generated memory points and dialogue")
            # For initial events, calculate dialogue turns based on actual generated dialogue
            dialogue_turn_num = len(dialogue.keys())
            print(f"[DEBUG] Actual dialogue turns for initial event: {dialogue_turn_num}")
            
            memory_points = event['dialogue_info']['memory_points']
            
            # Add system memory point annotations for init events
            for mp in memory_points:
                mp['memory_source'] = 'system'
            
            # Assign timestamps for memory points
            memory_points = assign_memory_point_timestamps(
                memory_points_data=memory_points,
                dialogue_data=dialogue,
                dialogue_turn_num=dialogue_turn_num
            )
        else:
            # Non-initial event, use validated memory points from generate_dialogue_for_event
            if validated_memory_points is not None:
                memory_points = validated_memory_points
                print(f"[DEBUG] Using validated memory points from generate_dialogue_for_event: {len(memory_points)} points")
            else:
                raise ValueError(f"Validated memory points missing for event {event.get('event_name')} ({event.get('event_type')})")
            
            # Assign timestamps for memory points
            memory_points = assign_memory_point_timestamps(
                memory_points_data=memory_points,
                dialogue_data=dialogue,
                dialogue_turn_num=dialogue_turn_num
            )
        
        # Merge all types of memory points (system memory points, secondary memory points, interference memory points)
        all_memory_points = memory_points.copy()
        
        # If there are interference memory points, add to memory point list
        if 'interference_memories' in locals() and interference_memories:
            # Get dialogue end time
            dialogue_turns = sorted(dialogue.keys(), key=lambda k: int(k.split('_')[-1]))
            end_timestamp = None
            if dialogue_turns:
                last_turn = dialogue_turns[-1]
                if dialogue[last_turn] and len(dialogue[last_turn]) > 0:
                    end_timestamp = dialogue[last_turn][-1].get("timestamp")
            
            # Re-number and ensure timestamp field for interference memory points
            for i, mp in enumerate(interference_memories):
                mp['index'] = len(all_memory_points) + i + 1
                if 'timestamp' not in mp or mp['timestamp'] == "Jan 15, 2025, 14:30:00":
                    mp['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
            all_memory_points.extend(interference_memories)
        
        # Ensure all memory points have correct format
        # Get dialogue end time
        dialogue_turns = sorted(dialogue.keys(), key=lambda k: int(k.split('_')[-1]))
        end_timestamp = None
        if dialogue_turns:
            last_turn = dialogue_turns[-1]
            if dialogue[last_turn] and len(dialogue[last_turn]) > 0:
                end_timestamp = dialogue[last_turn][-1].get("timestamp")
        
        for mp in all_memory_points:
            # Ensure memory_source field exists
            if 'memory_source' not in mp:
                mp['memory_source'] = 'secondary'
            
            # Ensure other necessary fields exist
            if 'memory_type' not in mp:
                mp['memory_type'] = 'persona'
            if 'is_update' not in mp:
                mp['is_update'] = 'False'
            if 'original_memories' not in mp:
                mp['original_memories'] = []
            if 'timestamp' not in mp or mp['timestamp'] == "Jan 15, 2025, 14:30:00":
                mp['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
            
            # Ensure importance field exists for all memory points
            if 'importance' not in mp:
                if mp.get('memory_source') == 'system':
                    mp['importance'] = 1.0
                elif mp.get('memory_source') == 'interference':
                    mp['importance'] = 1.0
                else:
                    mp['importance'] = 0.75
        
        # Calculate counts of different types of memory points
        system_memory_count = sum(1 for m in all_memory_points if m.get('memory_source') == 'system')
        secondary_memory_count = sum(1 for m in all_memory_points if m.get('memory_source') == 'secondary')
        interference_memory_count = sum(1 for m in all_memory_points if m.get('memory_source') == 'interference')
        update_memory_points_count = sum(1 for m in all_memory_points if m.get("is_update", "False") == "True")
        
        # Update event data - adapt to stage4 format, only keep one memory_points field
        event_data["dialogue_info"]["dialogue"] = dialogue
        event_data["dialogue_info"]["dialogue_turn_num"] = dialogue_turn_num
        event_data["dialogue_info"]["memory_points"] = all_memory_points
        event_data["dialogue_info"]["memory_points_count"] = len(all_memory_points)
        
        # Update current memory points and event list - ensure all memory points have correct format
        # Get dialogue end time
        dialogue_turns = sorted(dialogue.keys(), key=lambda k: int(k.split('_')[-1]))
        end_timestamp = None
        if dialogue_turns:
            last_turn = dialogue_turns[-1]
            if dialogue[last_turn] and len(dialogue[last_turn]) > 0:
                end_timestamp = dialogue[last_turn][-1].get("timestamp")
        
        for mp in memory_points:
            # Ensure memory_source field exists
            if 'memory_source' not in mp:
                mp['memory_source'] = 'secondary'
            
            # Ensure other necessary fields exist
            if 'memory_type' not in mp:
                mp['memory_type'] = 'persona'
            if 'is_update' not in mp:
                mp['is_update'] = 'False'
            if 'original_memories' not in mp:
                mp['original_memories'] = []
            if 'timestamp' not in mp or mp['timestamp'] == "Jan 15, 2025, 14:30:00":
                mp['timestamp'] = end_timestamp or "Jan 15, 2025, 14:30:00"
            
            # Ensure importance field exists for all memory points
            if 'importance' not in mp:
                if mp.get('memory_source') == 'system':
                    mp['importance'] = 1.0
                elif mp.get('memory_source') == 'interference':
                    mp['importance'] = 1.0
                else:
                    mp['importance'] = 0.75
        
        current_memory_points.extend(
            json.dumps(i, ensure_ascii=False) for i in memory_points
        )
        
        new_data["event_list"].append(event_data)
        
        # Format event time
        try:
            event_time = datetime.strptime(event['event_time'], "%Y-%m-%d").strftime("%b %d, %Y")
            current_events.append(
                f"  - [{event_time}]{event['event_name']}: {event['event_description']}"
            )
        except Exception as e:
            print(f"[ERROR] Event time formatting failed: {e}:{traceback.format_exc()}")
            current_events.append(
                f"  - [Unknown Date]{event['event_name']}: {event['event_description']}"
            )
    
    print("[DEBUG] Processing complete")
    
    final_token_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
    
    new_data["token_cost"] = final_token_cost
    
    return new_data





def process_all_users(regenerate: bool = True, test_mode: bool = False, test_event_count: int = None):
    """Process all user data"""
    input_file = config['file_paths']['input_file']
    output_file = config['file_paths']['output_file']
    
    print(f"Batch processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Regenerate mode: {'Yes' if regenerate else 'No'}")
    if test_mode:
        print(f"Test mode: Yes, processing {test_event_count} events per user")
    else:
        print(f"Test mode: No")
    
    # If in regenerate mode, delete previous output file
    if regenerate and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[DEBUG] Deleted previous output file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Error deleting output file: {e}:{traceback.format_exc()}")
    
    # If not in regenerate mode, check if output file exists and contains data
    if not regenerate and os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                existing_count = sum(1 for _ in reader)
            if existing_count > 0:
                print(f"[DEBUG] Output file already exists and contains {existing_count} data, skipping processing")
                return True
        except Exception as e:
            print(f"[ERROR] Error checking output file: {e}:{traceback.format_exc()}")
    
    try:
        # Get processed UUIDs
        existing_uuids = set()
        if os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                for item in reader:
                    if isinstance(item, dict) and 'uuid' in item:
                        existing_uuids.add(item['uuid'])
        
        all_user_data = []
        skipped_count = 0
        
        with jsonlines.open(input_file) as reader:
            for user_data in reader:
                user_uuid = user_data.get('uuid')
                if not user_uuid:
                    print("[DEBUG] Skipping user data with missing UUID")
                    skipped_count += 1
                    continue
                
                if user_uuid in existing_uuids:
                    print(f"[DEBUG] Skipping already processed user: {user_uuid}")
                    skipped_count += 1
                    continue
                
                all_user_data.append(user_data)
        
        def process_single_user_data(user_data):
            try:
                # Process user data
                result_data = process_single_user(user_data, test_mode, test_event_count)
                print(f"[DEBUG] Successfully processed user {user_data.get('uuid')}")
                return result_data
            except Exception as e:
                print(f"[ERROR] Error processing user {user_data.get('uuid')}: {e}")
                return None
        
        processed_results = parallel_process(all_user_data, process_single_user_data, "Stage5.1-DialogueGeneration")
        
        valid_results = [result for result in processed_results if result is not None]
        with jsonlines.open(output_file, 'w') as writer:
            for result in valid_results:
                writer.write(result)
        
        print(f"[DEBUG] Batch processing complete, processed {len(valid_results)} user data, skipped {skipped_count} users")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during batch processing: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate dialogue content')
    parser.add_argument('--regenerate', action='store_true', default=True, 
                       help='Whether to completely regenerate (default: True)')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip existing parts (mutually exclusive with --regenerate)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Enable test mode, randomly select a specified number of events to process')
    parser.add_argument('--test-event-count', type=int, 
                       help='Number of events to process per user in test mode')
    
    args = parser.parse_args()
    
    # Determine generation mode
    if args.skip_existing:
        regenerate = False
    else:
        regenerate = args.regenerate
    
    # Determine test mode
    test_mode = args.test_mode
    test_event_count = args.test_event_count
    
    # Validate test mode parameters
    if test_mode and test_event_count is None:
        print("[ERROR] Test mode requires --test-event-count parameter")
        exit(1)
    
    if test_mode and test_event_count <= 0:
        print("[ERROR] Test event count must be greater than 0")
        exit(1)
    
    print(f"Generation mode: {'Regenerate' if regenerate else 'Skip existing'}")
    if test_mode:
        print(f"Test mode: Yes, processing {test_event_count} events per user")
    else:
        print("Test mode: No")
    
    print("Interference memory point feature: Enabled")
    
    print("Processing all user data...")
    process_all_users(regenerate, test_mode, test_event_count)
