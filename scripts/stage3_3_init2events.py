from datetime import datetime, timedelta
import json
import os
import jsonlines
import traceback
from dotenv import load_dotenv
from typing import Dict, List

from llm_request import calculate_cumulative_cost
from parallel_utils import parallel_process

# Load environment variables
load_dotenv()

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE3']['substage3']


def _format_date(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _extract_profile(obj: Dict) -> Dict:
    # Compatible with two inputs: old (stage2_3) contains metadata.persona_profile, new (profile) top level is profile
    if 'profile' in obj:
        return obj['profile']
    return _safe_get(obj, ['metadata', 'persona_profile'], {}) or {}


def _extract_dynamic_init(dynamic_obj: Dict) -> Dict:
    # Dynamic items after stage2_2/2_3 have { init, evolution_path } structure
    if isinstance(dynamic_obj, dict) and 'init' in dynamic_obj:
        return dynamic_obj.get('init', {})
    return dynamic_obj or {}


def _read_token_cost_from_stage3_2(uuid: str) -> Dict:
    
    stage3_2_path = os.getenv('STAGE3_2_FILE_PATH', 'data/stage3_2_preference_events.jsonl')
    try:
        with jsonlines.open(stage3_2_path) as reader:
            for obj in reader:
                if obj.get('uuid') == uuid:
                    return obj.get('token_cost')
    except Exception as e:
        print(f"[DEBUG] Failed to read stage3_2 token cost: {e}:{traceback.format_exc()}")
    return None


def generate_initial_events(profile: Dict) -> List[Dict]:
    """Directly generate initial event list, no LLM call"""
    events = []
    today = _format_date(datetime.now())
    
    # Fixed information events
    fixed = profile.get('fixed', {})
    if fixed:
        events.append({
            "event_type": "initial_fixed",
            "event_name": "Initial Information - Fixed Profile",
            "event_time": today,
            "event_description": "Description of initial state of character's basic profile",
            "initial_fixed": fixed
        })
    
    # Dynamic information initial events - merge into one event
    dynamic = profile.get('dynamic', {})
    if dynamic:
        # Collect all dynamic initial values
        all_dynamic_init = {}
        for dynamic_type, dynamic_val in dynamic.items():
            init_val = _extract_dynamic_init(dynamic_val)
            if init_val:
                all_dynamic_init[dynamic_type] = init_val
        
        if all_dynamic_init:
            events.append({
                "event_type": "initial_dynamic",
                "event_name": "Initial Information - Dynamic Profile",
                "event_time": today,
                "event_description": "Description of initial state of character's dynamic information",
                "initial_dynamic": all_dynamic_init
            })
    
    # Preference initial events - merge into one event
    preferences = profile.get('preferences', {})
    if preferences:
        # Collect all preference initial values
        all_preference_init = {}
        for pref_type, pref_val in preferences.items():
            init_val = pref_val.get('init', pref_val)
            if init_val:
                all_preference_init[pref_type] = init_val
        
        if all_preference_init:
            events.append({
                "event_type": "initial_preference",
                "event_name": "Initial Information - Preference Profile",
                "event_time": today,
                "event_description": "Description of initial state of character's preference information",
                "initial_preference": all_preference_init
            })
    
    return events


def process_init_events(input_file: str, output_file: str, regenerate: bool = True):
    """Process all profiles, generate initial events and save"""
    print(f"Reading file: {input_file}")
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
    
    data: List[Dict] = []
    try:
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                data.append(obj)
        print(f"[DEBUG] Successfully read {input_file}, total {len(data)} entries")
    except Exception as e:
        print(f"[ERROR] Failed to read {input_file}: {e}:{traceback.format_exc()}")
        return

    def process_single_item(obj):
        uuid_key = obj.get('uuid')
        if not uuid_key:
            return None
        profile = _extract_profile(obj)
        print(f"[DEBUG] Processing profile: UUID={uuid_key}")
        
        current_file_cost = obj.get('token_cost')
        correct_previous_cost = _read_token_cost_from_stage3_2(uuid_key)
        
        previous_cost = correct_previous_cost if correct_previous_cost else current_file_cost
        if correct_previous_cost:
            print(f"[DEBUG] Using token cost from stage3_2 (correct previous stage)")
        else:
            print(f"[DEBUG] Stage3_2 cost not found, using current file cost")
        
        current_stage_cost = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "model": None,
            "pricing_available": False,
            "note": "No LLM calls in this stage"
        }
        
        token_cost = calculate_cumulative_cost(previous_cost, current_stage_cost)
        
        # Directly generate initial events
        event_list = generate_initial_events(profile)
        
        output_data = {
            "uuid": uuid_key,
            "profile": profile,
            "event_list": event_list,
            "metadata": {
                "profile": profile,
                "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "token_cost": token_cost
        }
        print(f"[DEBUG] Successfully generated initial event list, total {len(event_list)} events")
        return output_data
    
    processed_results = parallel_process(data, process_single_item, "Stage3.3-Init2Events")
    
    with jsonlines.open(output_file, 'w') as writer:
        for result in processed_results:
            if result is not None:
                writer.write(result)


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate initial event persona information')
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
    
    input_file = os.getenv('STAGE2_3_FILE_PATH', 'data/stage2_3_profile2skeleton.jsonl')
    output_file = os.getenv('STAGE3_3_FILE_PATH', 'data/stage3_3_init_events.jsonl')
    if not os.path.exists(input_file):
        print(f"[ERROR] File does not exist: {input_file}")
        raise SystemExit(1)
    process_init_events(input_file, output_file, regenerate)



