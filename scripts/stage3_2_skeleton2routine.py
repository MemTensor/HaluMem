from datetime import datetime
import json
import os
import traceback
import jsonlines
import logging
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import random
import copy  # Add copy module for deep copy

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

# Load environment variables
load_dotenv()

RETRY_TIMES = int(os.getenv('RETRY_TIMES'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER'))


with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE3']['substage2']

def read_life_skeleton_from_file(file_path: str) -> List[Dict]:
    """Read life skeleton, persona_profile and preference_stage_info, and corresponding UUID from file"""
    life_skeletons_with_uuid = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if 'life_skeleton' in obj and 'uuid' in obj:
                life_skeletons_with_uuid.append({
                    'life_skeleton': obj['life_skeleton'],
                    'persona_profile': obj.get('metadata', {}).get('persona_profile', {}),
                    'preference_stage_info': obj.get('metadata', {}).get('preference_stage_info', {}),
                    'uuid': obj['uuid'],
                    'metadata': obj.get('metadata', {})
                })
            elif 'profile' in obj and 'uuid' in obj:
                profile = obj['profile']
                fixed = profile.get('fixed', {})
                life_goal = fixed.get('life_goal', 'Life Goal')
                career_events = profile.get('career_events', [])
                life_skeletons_with_uuid.append({
                    'life_skeleton': {
                        'life_goal': life_goal,
                        'career_events': career_events
                    },
                    'persona_profile': profile,
                    'preference_stage_info': {},
                    'uuid': obj['uuid'],
                    'metadata': obj.get('metadata', {})
                })
    return life_skeletons_with_uuid


def _read_preference_types_order_from_stage1_3(uuid: str) -> List[str]:
    """Read preference type order for given UUID from stage1_3_preferences.jsonl."""
    stage1_3_path = os.getenv('STAGE1_3_FILE_PATH', 'data/stage1_3_preferences.jsonl')
    try:
        with jsonlines.open(stage1_3_path) as reader:
            for obj in reader:
                if obj.get('uuid') == uuid:
                    prefs = obj.get('profile', {}).get('preferences', {})
                    if not prefs:
                        raise ValueError(f"Preferences missing for uuid {uuid} in stage1_3 file {stage1_3_path}")
                    # Maintain JSON original key order
                    return list(prefs.keys())
    except Exception as e:
        raise RuntimeError(f"Failed to read stage1_3 preference order: {e}:{traceback.format_exc()}")
    raise ValueError(f"Preference order for uuid {uuid} not found in stage1_3 file {stage1_3_path}")

def _read_token_cost_from_stage3_1(uuid: str) -> Dict:
    """
    Read token_comst information from the output file of stage3_1
    This is the correct previous document in terms of process, ensuring the correctness of cost accumulation
    """
    stage3_1_path = os.getenv('STAGE3_1_FILE_PATH', 'data/stage3_1_narrative_arc.jsonl')
    try:
        with jsonlines.open(stage3_1_path) as reader:
            for obj in reader:
                if obj.get('uuid') == uuid:
                    return obj.get('metadata', {}).get('token_cost')
    except Exception as e:
        print(f"[DEBUG] Failed to read stage3_1 token cost: {e}:{traceback.format_exc()}")
    return None

def generate_preference_event_template(events: List[Dict]) -> str:
    """Generate preference evolution event large event list template (without time), customized placeholders by direction."""
    def tmpl_for_direction(direction: str) -> Dict:
        # Add: only add one new memory point; before is empty list, after only contains new point
        if direction == "Add":
            return {
                "before_preference": {"memory_points": []},
                "after_preference": {"memory_points": [
                    {
                        "type": "like or dislike (choose one)",
                        "type_description": "Chinese description consistent with type",
                        "specific_item": "New specific item",
                        "reason": "Reason for addition"
                    }
                ]}
            }
        # Modify: only select and modify one existing memory point; before/after each only contains that one
        if direction == "Modify":
            return {
                "before_preference": {"memory_points": [
                    {
                        "type": "Original value",
                        "type_description": "Original value",
                        "specific_item": "Original value",
                        "reason": "Original value"
                    }
                ]},
                "after_preference": {"memory_points": [
                    {
                        "type": "Can change from dislike to like, or remain unchanged",
                        "type_description": "Chinese description consistent with type",
                        "specific_item": "Modified specific item",
                        "reason": "Modified reason"
                    }
                ]}
            }
        # Delete: only select one existing memory point; after retains that one and marks as deleted
        if direction == "Delete":
            # Create base memory point object
            base_memory_point = {
                "type": "Original value",
                "type_description": "Original value",
                "specific_item": "Original value",
                "reason": "Original value"
            }
            
            # Use deep copy to ensure before and after are independent objects
            before_memory_point = copy.deepcopy(base_memory_point)
            after_memory_point = copy.deepcopy(base_memory_point)
            after_memory_point["deleted"] = True
            
            return {
                "before_preference": {"memory_points": [before_memory_point]},
                "after_preference": {"memory_points": [after_memory_point]}
            }
        raise ValueError(f"Unsupported update_direction '{direction}' in preference events")

    items = []
    for ev in events:
        direction = ev.get('update_direction', '')
        base = tmpl_for_direction(direction)
        before_pref = json.dumps(base["before_preference"], ensure_ascii=False)
        after_pref = json.dumps(base["after_preference"], ensure_ascii=False)

        items.append(f"""
    {{
      "event_id": {ev.get('event_id', 0)},
      "preference_type": "{ev.get('preference_type', '')}",
      "step": {ev.get('step', 1)},
      "main_conflict": "Main conflict/compromise faced by this preference change (must be specific)",
      "type_to_update": "{ev.get('preference_type', '')}",
      "update_direction": "{direction}",
      "before_preference": {before_pref},
      "update_reason": "Reason causing the change (must be specific, pointing to event trigger mechanism)",
      "after_preference": {after_pref},
      "changed_index": "Index of changed memory point in memory_points array (starting from 0, null if new addition)",
      "event_description": "More vivid, complete event narrative: environment/character/motivation/action/conflict/turning point/result"
    }}""")
    return "[" + ",".join(items) + "\n]"

def build_preference_evolution_events(persona_profile: Dict, ordered_types: List[str]) -> List[Dict]:
    """Expand each preference's each evolution into separate event placeholders.
    Use init and evolution_path from persona_profile['profile']['preferences'].
    before/after filled by LLM, here only provide placeholders and direction.
    """
    events: List[Dict] = []
    preferences = (persona_profile or {}).get('preferences') or (persona_profile or {}).get('profile', {}).get('preferences', {})
    if not preferences:
        raise ValueError("Persona profile missing 'preferences' data for preference evolution")
    if not any(pref.get('evolution_path') for pref in preferences.values()):
        raise ValueError("No preference evolution steps found; cannot generate preference events")
    eid = 1
    # Confirm preference type order: prioritize stage1_3 order; otherwise use preference key sorting; raise when invalid
    if ordered_types:
        pref_types_iter = ordered_types
    else:
        try:
            pref_types_iter = sorted(list(preferences.keys()))
        except Exception as e:
            raise ValueError(f"Invalid preference keys encountered while building evolution events: {e}:{traceback.format_exc()}")

    for pref_type in pref_types_iter:
        if pref_type not in preferences:
            raise ValueError(f"Preference type '{pref_type}' missing in persona preferences")
        pref_obj = preferences.get(pref_type, {})
        evo = pref_obj.get('evolution_path', []) or []
        # Sort by step in ascending order, ensure same preference different stages are consecutive and ordered
        try:
            evo = sorted(evo, key=lambda x: x.get('step', 0))
        except Exception as e:
            raise ValueError(f"Invalid evolution path for preference '{pref_type}': {e}:{traceback.format_exc()}")
        for step_item in evo:
            step_num = step_item.get('step', 1)
            direction = step_item.get('direction', '')
            direction_suffix = f"{direction} " if direction else ""
            event_name = f"{pref_type} - Step {step_num} {direction_suffix}Evolution".strip()
            events.append({
                'event_id': eid,
                'preference_type': pref_type,
                'step': step_num,
                'update_direction': direction,
                'event_name': event_name,
                'before_preference': {},
                'after_preference': {}
            })
            eid += 1
    return events


def group_events_by_preference_type(events: List[Dict]) -> Dict[str, List[Dict]]:
    """Group events by preference type"""
    grouped_events = {}
    for event in events:
        pref_type = event.get('preference_type')
        if pref_type is None:
            raise ValueError(f"Preference event missing 'preference_type': {event}")
        if pref_type not in grouped_events:
            grouped_events[pref_type] = []
        grouped_events[pref_type].append(event)
    return grouped_events


def generate_single_preference_template(preference_type: str, events: List[Dict]) -> str:
    """Generate a JSON template for a single preference type"""
    events_template = []
    for event in events:
        if not event.get('event_name') or not isinstance(event.get('event_name'), str):
            raise ValueError(f"Preference event missing event_name: {event}")
        event_template = {
            "event_id": event.get('event_id', 0),
            "preference_type": preference_type,
            "step": event.get('step', 1),
            "event_name": event.get('event_name'),
            "type_to_update": preference_type,
            "update_direction": event.get('update_direction', ''),
            "before_preference": {},
            "update_reason": f"Reason for {preference_type} evolution step {event.get('step', 1)}",
            "after_preference": {},
            "changed_index": None,
            "main_conflict": f"Main conflict for {preference_type} step {event.get('step', 1)}",
            "event_description": f"Event description for {preference_type} step {event.get('step', 1)}"
        }
        events_template.append(event_template)
    
    return json.dumps(events_template, ensure_ascii=False, indent=2)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def process_single_preference_type(
    preference_type: str,
    preference_events: List[Dict],
    life_skeleton: Dict,
    persona_profile: Dict,
    current_cost: Dict
) -> Tuple[List[Dict], Dict]:
    """
    Handling all evolutionary events of a single preference type
    """
    
    preferences = (persona_profile or {}).get('preferences') or (persona_profile or {}).get('profile', {}).get('preferences', {})
    current_preference = preferences.get(preference_type, {})
    
    events_template = generate_single_preference_template(preference_type, preference_events)
    
    user_content = f"""
<Ultimate Life Goal>
{life_skeleton.get('life_goal', 'Life Goal')}
</Ultimate Life Goal>

<Initial Preferences (for filling before_preference)>
{json.dumps(preferences, ensure_ascii=False, indent=2)}
</Initial Preferences (for filling before_preference)>

<Current Preference Type to Process>
{preference_type}
</Current Preference Type to Process>

<Current Preference Evolution Path (for reference only, do not modify)>
{json.dumps(current_preference.get('evolution_path', []), ensure_ascii=False, indent=2)}
</Current Preference Evolution Path (for reference only, do not modify)>

<Response Template>
{{
  "event_list": {events_template}
}}
</Response Template>
"""

    with open('prompts/stage3_2_skeleton2routine.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    try:
        print(f"[DEBUG] Processing preference type: {preference_type} ({len(preference_events)} events)")

        json_markers = [
            "Generated preference events", "Final JSON", "Complete JSON", "Correction result"
        ]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=json_markers)
        print(f"[DEBUG] Raw LLM response for {preference_type}: {raw_response[:200]}...")
        
        token_cost = calculate_cumulative_cost(current_cost, cost_info)
        
        current_stage_cost = cost_info
        cumulative_cost = token_cost.get('cumulative', {})
        print(f"[DEBUG] {preference_type} - Input: {current_stage_cost.get('input_tokens', 'N/A')}, "
              f"Output: {current_stage_cost.get('output_tokens', 'N/A')}, "
              f"Cost: ${current_stage_cost.get('total_cost_usd', 'N/A')}")
        print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
              f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        try:
            preference_result = _extract_json_from_content(raw_response, json_markers)
            event_list = preference_result.get('event_list', [])
            print(f"[DEBUG] Parsed event_list for {preference_type}: {event_list}")
            for ev in event_list:
                if not isinstance(ev.get('event_name'), str) or not ev.get('event_name').strip():
                    raise ValueError(f"LLM output missing event_name for preference type {preference_type}: {ev}")
            print(f"[DEBUG] Successfully parsed JSON for {preference_type}: {len(event_list)} events")
            return event_list, token_cost
        except ValueError as e:
            print(f"[DEBUG] JSON parsing error for {preference_type}: {e}:{traceback.format_exc()}")
            raise ValueError(f"Failed to parse JSON for preference type {preference_type}: {e}:{traceback.format_exc()}")

    except Exception as e:
        print(f"[DEBUG] Error processing preference type {preference_type}: {e}:{traceback.format_exc()}")
        raise Exception(f"Failed to process preference type {preference_type}: {e}:{traceback.format_exc()}")


def _time_diff_months(time1: str, time2: str) -> int:
    """Calculate month difference between two times"""
    try:
        from datetime import datetime
        date1 = datetime.strptime(time1, "%Y-%m-%d")
        date2 = datetime.strptime(time2, "%Y-%m-%d")
        months_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
        return months_diff
    except Exception as e:
        print(f"[ERROR] Time parsing error: {e}:{traceback.format_exc()}, time1: {time1}, time2: {time2}")
        return 0

def _generate_random_time_in_range(start_time: str, end_time: str) -> str:
    """Generate random time point within time range"""
    try:
        from datetime import datetime, timedelta
        
        start_date = datetime.strptime(start_time, "%Y-%m-%d")
        end_date = datetime.strptime(end_time, "%Y-%m-%d")
        
        # Calculate time difference (days)
        time_diff = (end_date - start_date).days
        
        if time_diff <= 0:
            return start_time
        
        # Generate random day offset
        random_days = random.randint(0, time_diff)
        random_date = start_date + timedelta(days=random_days)
        
        result_time = random_date.strftime("%Y-%m-%d")
        return result_time
        
    except Exception as e:
        print(f"[ERROR] Error generating random time: {e}:{traceback.format_exc()}")
        return start_time

def generate_preference_events_doc(life_skeleton: Dict, persona_profile: Dict, persona_uuid: str, previous_cost: Dict = None) -> Dict:
    """Generate large event list template based on preferences evolution using incremental processing by preference type."""
    # Preference source prioritizes persona_profile (new structure from stage2_2/2_3)
    preferences = (persona_profile or {}).get('preferences') or (persona_profile or {}).get('profile', {}).get('preferences', {})
    if not preferences:
        raise ValueError("No preferences information found in persona profile")

    # Preference type order: based on stage1_3 results
    ordered_types = _read_preference_types_order_from_stage1_3(persona_uuid)
    # Build event placeholders (same preference different stages arranged consecutively)
    events = build_preference_evolution_events(persona_profile, ordered_types)
    print(f"[DEBUG] Preference evolution event count: {len(events)}")

    events_by_preference = group_events_by_preference_type(events)
    print(f"[DEBUG] Preference types to process: {list(events_by_preference.keys())}")
    
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }
    
    all_event_results = []
    
    if ordered_types:
        preference_types_to_process = [pt for pt in ordered_types if pt in events_by_preference]
        for pt in events_by_preference.keys():
            if pt not in preference_types_to_process:
                preference_types_to_process.append(pt)
    else:
        preference_types_to_process = list(events_by_preference.keys())
    
    for pref_type_index, preference_type in enumerate(preference_types_to_process):
        try:
            preference_events = events_by_preference[preference_type]
            print(f"[DEBUG] Processing preference type {pref_type_index + 1}/{len(preference_types_to_process)}: {preference_type} ({len(preference_events)} events)")
            
            event_results, event_cost = process_single_preference_type(
                preference_type=preference_type,
                preference_events=preference_events,
                life_skeleton=life_skeleton,
                persona_profile=persona_profile,
                current_cost=previous_cost if pref_type_index == 0 else None
            )
            
            if event_cost and "current_stage" in event_cost:
                current_stage_cost = event_cost["current_stage"]
                current_stage_total_cost["input_tokens"] += current_stage_cost.get("input_tokens", 0)
                current_stage_total_cost["output_tokens"] += current_stage_cost.get("output_tokens", 0)
                current_stage_total_cost["total_tokens"] += current_stage_cost.get("total_tokens", 0)
                current_stage_total_cost["total_cost_usd"] += current_stage_cost.get("total_cost_usd", 0.0)
                if current_stage_total_cost["model"] is None:
                    current_stage_total_cost["model"] = current_stage_cost.get("model")
                current_stage_total_cost["pricing_available"] = current_stage_cost.get("pricing_available", False)
            
            all_event_results.extend(event_results)
            
            print(f"[DEBUG] Successfully processed preference type: {preference_type}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process preference type {preference_type}: {e}:{traceback.format_exc()}")
            raise Exception(f"Failed to process preference type {preference_type}: {e}:{traceback.format_exc()}")
    
    final_token_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
    
    print(f"[DEBUG] All preference types processed successfully. Total preference types: {len(preference_types_to_process)}")
    print(f"[DEBUG] Final cost - Input: {current_stage_total_cost.get('input_tokens', 'N/A')}, "
          f"Output: {current_stage_total_cost.get('output_tokens', 'N/A')}, "
          f"Cost: ${current_stage_total_cost.get('total_cost_usd', 'N/A')}")
    
    return {
        "event_list": all_event_results,
        "raw_query": {},
        "raw_response": "",
        "token_cost": final_token_cost
    }

def process_life_skeletons(file_path: str, output_file: str, regenerate: bool = True):
    """Process all life skeletons, generate preference evolution events and save"""
    print(f"Reading file: {file_path}")
    print(f"Output file: {output_file}")
    print(f"Regeneration mode: {regenerate}")
    
    # If not regeneration mode, check if output file already exists and contains data
    if not regenerate and os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                existing_count = sum(1 for _ in reader)
            if existing_count > 0:
                print(f"[DEBUG] Output file already exists and contains {existing_count} data entries, skipping processing")
                return True
        except Exception as e:
            print(f"[DEBUG] Error checking output file: {e}:{traceback.format_exc()}")
    
    life_skeletons_with_uuid = read_life_skeleton_from_file(file_path)
    
    def process_single_item(item):
        life_skeleton = item['life_skeleton']
        persona_profile = item.get('persona_profile', {})
        persona_uuid = item['uuid']
        
        print(f"[DEBUG] Processing life skeleton: {life_skeleton.get('life_goal', 'Unknown')} (UUID: {persona_uuid})")
        
        current_file_cost = item.get('metadata', {}).get('token_cost')
        correct_previous_cost = _read_token_cost_from_stage3_1(persona_uuid)
        
        previous_cost = correct_previous_cost if correct_previous_cost else current_file_cost
        if correct_previous_cost:
            print(f"[DEBUG] Using token cost from stage3_1 (correct previous stage)")
        else:
            print(f"[DEBUG] Stage3_1 cost not found, using current file cost")
        
        # Call generation function
        result = generate_preference_events_doc(life_skeleton, persona_profile, persona_uuid, previous_cost)
        
        if result:
            output_data = {
                "uuid": persona_uuid,
                "event_list": result.get('event_list', []),
                "profile": persona_profile,
                "metadata": {
                    "life_skeleton": life_skeleton,
                    "persona_profile": persona_profile,
                    "raw_query": result.get('raw_query', {}),
                    "raw_response": result.get('raw_response', {}),
                    "parse_error": result.get('parse_error', None),
                    "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "token_cost": result.get('token_cost')
            }
            if result.get('event_list'):
                print(f"[DEBUG] Successfully generated preference evolution event list")
            else:
                print(f"[DEBUG] Failed to generate preference evolution events, but saved original response")
            return output_data
        else:
            print(f"[DEBUG] Completely failed to generate preference evolution events")
            return None
    
    processed_results = parallel_process(life_skeletons_with_uuid, process_single_item, "Stage3.2-Skeleton2Routine")
    
    with jsonlines.open(output_file, 'w') as writer:
        for result in processed_results:
            if result is not None:
                writer.write(result)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate preference evolution event persona information')
    parser.add_argument('--regenerate', action='store_true', default=True, 
                       help='Whether to completely regenerate (default: True)')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip existing parts (mutually exclusive with --regenerate)')
    
    args = parser.parse_args()
    
    # Determine generation mode
    if args.skip_existing:
        regenerate = False
    else:
        regenerate = args.regenerate
    
    print(f"Generation mode: {'Regenerate' if regenerate else 'Skip existing'}")
    
    input_file = os.getenv('STAGE2_3_FILE_PATH', 'data/stage2_3_profile2skeleton.jsonl')  # Read data from stage2_3
    output_file = os.getenv('STAGE3_2_FILE_PATH', 'data/stage3_2_preference_events.jsonl')  # Current step output file
    process_life_skeletons(input_file, output_file, regenerate)
