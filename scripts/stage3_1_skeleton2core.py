from datetime import datetime
import json
import os
import traceback
import jsonlines
import logging
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import random

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
    config = config['STAGE3']['substage1']

def read_life_skeleton_from_file(file_path: str) -> List[Dict]:
    """Read life skeleton, persona_profile and corresponding UUID from file.
    Compatible with two input formats:
    - Old format: top level contains life_skeleton, and metadata.persona_profile is complete persona
    - New format: top level contains profile (i.e., complete persona), where career_events is located at profile.career_events
    """
    life_skeletons_with_uuid = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if 'life_skeleton' in obj and 'uuid' in obj:
                persona_profile = obj.get('metadata', {}).get('persona_profile')
                life_skeletons_with_uuid.append({
                    'life_skeleton': obj['life_skeleton'],
                    'persona_profile': persona_profile,
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
                    'uuid': obj['uuid'],
                    'metadata': obj.get('metadata', {})
                })
    return life_skeletons_with_uuid

def sample_stage_count_for_career_events(career_events: List[Dict]) -> List[int]:
    """Sample stage count for each core event"""
    event_stage_count_list = []
    event_stage_count_range = config['arc_stage_count_range']
    
    for event in career_events:
        event_stage_count = random.randint(event_stage_count_range[0], event_stage_count_range[1])
        event_stage_count_list.append(event_stage_count)
    
    return event_stage_count_list

def assign_dynamic_content_to_stages(career_event: Dict, stage_count: int) -> List[Dict]:
    """Assign core event's dynamic content to various stages.
    Template adjustment: dynamic_updates placed after main_conflict, and each update has the following structure:
      - type_to_update: Dynamic type to update (fixed value, must not modify)
      - update_direction: Update direction (fixed value, must not modify)
      - before_dynamic: Complete value before update (filled by model)
      - update_reason: Update reason (filled by model)
      - after_dynamic: Complete value after update (filled by model)
      - changed_keys: List of affected keys (filled by model)
    Quantity strictly equals the number of items assigned to that stage.
    """
    assigned_content = career_event.get('assigned_dynamic_content', [])
    stage_assignments = []

    # Initialize stage containers
    for stage_id in range(1, stage_count + 1):
        stage_assignments.append({
            "stage_id": stage_id,
            "dynamic_updates": []
        })

    if not assigned_content:
        return stage_assignments

    # Try to evenly distribute dynamic items already assigned to events to stages
    for i, content in enumerate(assigned_content):
        target_stage = (i % stage_count) + 1
        stage_info = content.get("stage_info", {}) or {}
        direction = stage_info.get("stage_type") or stage_info.get("evolution_info", {}).get("direction", "")
        stage_assignments[target_stage - 1]["dynamic_updates"].append({
            "type_to_update": content.get("base_type"),
            "update_direction": direction,
            "before_dynamic": {},
            "update_reason": "",
            "after_dynamic": {},
            "changed_keys": []
        })

    return stage_assignments


def generate_career_event_stage_template(event: Dict, stage_count: int, stage_assignments: List[Dict]) -> str:
    """Generate core event stage template (dynamic_updates immediately follows main_conflict, structure has six elements)."""
    stages_template = ""

    for stage_id in range(1, stage_count + 1):
        stage_assignment = stage_assignments[stage_id - 1]
        dynamic_updates = stage_assignment["dynamic_updates"]
        dynamic_updates_template = json.dumps(dynamic_updates, ensure_ascii=False, indent=10)

        stages_template += f"""    {{
      "stage_id": {stage_id},
      "stage_name": "Stage Name",
      "time_point": "Stage time point, format as YYYY-MM-DD",
      "main_conflict": "Main core conflict",
      "dynamic_updates": {dynamic_updates_template},
      "stage_description": "Stage description",
      "stage_result": "Stage result"
    }}"""
        
        if stage_id < stage_count:
            stages_template += ",\n"

    return stages_template

def generate_narrative_arc_json_template(career_events: List[Dict], event_stage_count_list: List[int], life_skeleton: Dict) -> str:
    """Generate narrative arc JSON template (strict keys)."""
    events_template = ""

    for i, event in enumerate(career_events):
        event_stage_count = event_stage_count_list[i]
        if 'event_start_time' not in event or 'event_end_time' not in event:
            raise ValueError(f"Career event missing required time fields: {event}")
        stage_assignments = assign_dynamic_content_to_stages(event, event_stage_count)
        stages_template = generate_career_event_stage_template(event, event_stage_count, stage_assignments)

        events_template += f"""    {{
      "event_id": {event.get('event_id', i + 1)},
      "event_name": "{event.get('event_name', 'Event Name')}",
      "event_start_time": "{event['event_start_time']}",
      "event_end_time": "{event['event_end_time']}",
      "user_age": {event.get('user_age', 0)},
      "event_description": "{event.get('event_description', 'Detailed description of event background, process, and results')}",
      "event_result": "{event.get('event_result', 'Specific results of event')}",
      "event_stage_count": {event_stage_count},
      "narrative_arc": [
{stages_template}
      ]
    }}"""
        
        if i < len(career_events) - 1:
            events_template += ",\n"

    json_template = f"""
{{
  "life_goal": "{life_skeleton.get('life_goal', 'Life Goal')}",
  "career_events": [
{events_template}
  ]
}}"""

    return json_template


def generate_single_event_json_template(current_event: Dict, event_stage_count: int, stage_assignments: List[Dict], life_skeleton: Dict) -> str:
    """Generate JSON template for a single event"""
    event_name = current_event.get('event_name', 'Unknown Event')
    
    stages_template = []
    for stage_idx in range(event_stage_count):
        
        if stage_idx < len(stage_assignments):
            stage_assignment = stage_assignments[stage_idx]
            dynamic_updates = stage_assignment.get('dynamic_updates', [])
        else:
            dynamic_updates = []
        
        stage_template = {
            "stage_name": "Meaningful stage name describing this phase",
            "time_point": "Stage time point, format as YYYY-MM-DD",
            "stage_description": f"Description for {event_name} Stage {stage_idx + 1}",
            "stage_result": f"Result for {event_name} Stage {stage_idx + 1}",
            "dynamic_updates": []
        }
        
        for update in dynamic_updates:
            update_template = {
                "type_to_update": update.get('type_to_update', ''),
                "update_direction": update.get('update_direction', ''),
                "before_dynamic": {},
                "update_reason": f"Reason for {update.get('type_to_update', '')} update in {event_name} Stage {stage_idx + 1}",
                "after_dynamic": {},
                "changed_keys": []
            }
            stage_template["dynamic_updates"].append(update_template)
        
        stages_template.append(stage_template)
    
    json_template = f"""{{
  "event_name": "{event_name}",
  "event_description": "Detailed description for {event_name}",
  "event_result": "Final result for {event_name}",
  "event_start_time": "YYYY-MM-DD HH:MM:SS",
  "event_end_time": "YYYY-MM-DD HH:MM:SS",
  "stages": {json.dumps(stages_template, ensure_ascii=False, indent=2)}
}}"""
    
    return json_template


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _validate_event_time_fields(event_data: Dict, event_index: int) -> None:
    """Ensure event_start_time and event_end_time exist and follow YYYY-MM-DD HH:MM:SS."""
    required_fields = ['event_start_time', 'event_end_time']
    missing = [
        fld for fld in required_fields
        if fld not in event_data or not isinstance(event_data.get(fld), str) or not event_data.get(fld).strip()
    ]
    if missing:
        raise ValueError(f"Event {event_index + 1} missing required fields {missing}: {event_data}")

    for fld in required_fields:
        value = event_data.get(fld, '').strip()
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            raise ValueError(
                f"Event {event_index + 1} field '{fld}' has invalid format (expected YYYY-MM-DD HH:MM:SS): {value}"
            ) from exc


def process_single_event(
    current_event: Dict, 
    event_index: int,
    event_stage_count: int,
    stage_assignments: Dict,
    historical_context: List[Dict],
    life_skeleton: Dict,
    persona_profile: Dict,
    current_cost: Dict
) -> Tuple[Dict, Dict]:
    
    template_string = generate_single_event_json_template(current_event, event_stage_count, stage_assignments, life_skeleton)
    
    historical_context_str = ""
    if historical_context:
        historical_context_str = f"""
<Historical Context (Previously Processed Events)>
{json.dumps(historical_context, ensure_ascii=False, indent=2)}
</Historical Context (Previously Processed Events)>
"""
    
    user_content = f"""
<Ultimate Life Goal>
{life_skeleton.get('life_goal', 'Life Goal')}
</Ultimate Life Goal>

<Initial Dynamic (for filling before_dynamic)>
{json.dumps((persona_profile or {}).get('dynamic', {}), ensure_ascii=False, indent=2)}
</Initial Dynamic (for filling before_dynamic)>

<Initial Character Persona>
{json.dumps(life_skeleton.get('persona_update_dict', {}).get('0', {}), ensure_ascii=False, indent=2)}
</Initial Character Persona>
{historical_context_str}
<Current Event to Process>
{json.dumps(current_event, ensure_ascii=False, indent=2)}
</Current Event to Process>

<Current Event Stage Count>
{event_stage_count}
</Current Event Stage Count>

<Current Event Stage Assignment Information>
{json.dumps(stage_assignments, ensure_ascii=False, indent=2)}
</Current Event Stage Assignment Information>

<Response Template>
{template_string}
</Response Template>
"""

    with open('prompts/stage3_1_skeleton2core.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    try:
        print(f"[DEBUG] Processing event {event_index + 1}: {current_event.get('event_name', 'Unknown')}")

        json_markers = [
            "Generated narrative arc", "Final JSON", "Complete JSON", "Correction result"
        ]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=json_markers)
        
        token_cost = calculate_cumulative_cost(current_cost, cost_info)
        
        current_stage_cost = cost_info
        cumulative_cost = token_cost.get('cumulative', {})
        print(f"[DEBUG] Event {event_index + 1} - Input: {current_stage_cost.get('input_tokens', 'N/A')}, "
              f"Output: {current_stage_cost.get('output_tokens', 'N/A')}, "
              f"Cost: ${current_stage_cost.get('total_cost_usd', 'N/A')}")
        print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
              f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        try:
            event_result = _extract_json_from_content(raw_response, json_markers)
            _validate_event_time_fields(event_result, event_index)
            print(f"[DEBUG] Successfully parsed JSON for event {event_index + 1}")
            return event_result, token_cost
        except ValueError as e:
            print(f"[DEBUG] JSON parsing error for event {event_index + 1}: {e}:{traceback.format_exc()}")
            raise ValueError(f"Failed to parse JSON for event {event_index + 1}: {e}:{traceback.format_exc()}")

    except Exception as e:
        print(f"[DEBUG] Error processing event {event_index + 1}: {e}:{traceback.format_exc()}")
        raise Exception(f"Failed to process event {event_index + 1}: {e}:{traceback.format_exc()}")


def generate_narrative_arc(life_skeleton: Dict, persona_profile: Dict = None, previous_cost: Dict = None) -> Dict:
    """Generate character's narrative arc using incremental processing"""
    # Only process core events (career development)
    career_events = life_skeleton.get('career_events', [])
    
    if not career_events:
        print("[DEBUG] No core events found, skipping processing")
        return {}
    
    # 1. Program samples stage count
    event_stage_count_list = sample_stage_count_for_career_events(career_events)
    print(f"[DEBUG] Event stage count: {event_stage_count_list}")
    
    # 2. Assign dynamic content to stages for each event
    all_stage_assignments = []
    for i, event in enumerate(career_events):
        stage_count = event_stage_count_list[i]
        stage_assignments = assign_dynamic_content_to_stages(event, stage_count)
        all_stage_assignments.append(stage_assignments)
        print(f"[DEBUG] Event {i+1} stage assignment: {stage_assignments}")
    
    # 3. initialize cost
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }
    
    # 4. process events
    all_event_results = []
    historical_context = []
    
    for event_index, event in enumerate(career_events):
        try:
            print(f"[DEBUG] Processing event {event_index + 1}/{len(career_events)}: {event.get('event_name', 'Unknown')}")
            
            # process single event
            event_result, event_cost = process_single_event(
                current_event=event,
                event_index=event_index,
                event_stage_count=event_stage_count_list[event_index],
                stage_assignments=all_stage_assignments[event_index],
                historical_context=historical_context,
                life_skeleton=life_skeleton,
                persona_profile=persona_profile,
                current_cost=previous_cost if event_index == 0 else None
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
            
            all_event_results.append(event_result)
            historical_context.append(event_result)
            
            print(f"[DEBUG] Successfully processed event {event_index + 1}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process event {event_index + 1}: {e}:{traceback.format_exc()}")
            raise Exception(f"Failed to process event {event_index + 1}: {e}:{traceback.format_exc()}")
    
    # 5. Merge all event results into their original format
    merged_narrative_arc = {
        "career_events": all_event_results
    }
    
    # 6. Calculate the final cumulative cost
    final_token_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
    
    print(f"[DEBUG] All events processed successfully. Total events: {len(career_events)}")
    print(f"[DEBUG] Final cost - Input: {current_stage_total_cost.get('input_tokens', 'N/A')}, "
          f"Output: {current_stage_total_cost.get('output_tokens', 'N/A')}, "
          f"Cost: ${current_stage_total_cost.get('total_cost_usd', 'N/A')}")
    
    return {
        "event_stage_count_list": event_stage_count_list,
        "stage_assignments": all_stage_assignments,
        "narrative_arc": merged_narrative_arc,
        "raw_query": {},
        "raw_response": "",
        "token_cost": final_token_cost
    }

def process_life_skeletons(file_path: str, output_file: str, regenerate: bool = True):
    """Process all life skeletons, generate narrative arc and save"""
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
        persona_profile = item.get('persona_profile') or {}
        persona_uuid = item['uuid']
        
        print(f"[DEBUG] Processing life skeleton: {life_skeleton.get('life_goal', 'Life Goal')} (UUID: {persona_uuid})")
        
        previous_cost = item.get('metadata', {}).get('token_cost')
        
        # Call generate_narrative_arc function, which returns narrative_arc and constructed information
        result = generate_narrative_arc(life_skeleton, persona_profile, previous_cost)
        
        if result:
            # Build new output format，include token_cost
            output_data = {
                "uuid": persona_uuid,
                "narrative_arc": result.get('narrative_arc', {}),
                "profile": persona_profile,
                "life_skeleton": life_skeleton,
                "metadata": {
                    "event_stage_count_list": result.get('event_stage_count_list', []),
                    "stage_assignments": result.get('stage_assignments', []),
                    "life_skeleton": life_skeleton,
                    "raw_query": result.get('raw_query', {}),
                    "raw_response": result.get('raw_response', {}),
                    "parse_error": result.get('parse_error', None),
                    "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "token_cost": result.get('token_cost')
            }
            if result.get('narrative_arc'):
                print(f"[DEBUG] Successfully generated narrative arc for {life_skeleton.get('life_goal', 'Life Goal')}")
            else:
                print(f"[DEBUG] Failed to generate narrative arc, but saved original response for {life_skeleton.get('life_goal', 'Life Goal')}")
            return output_data
        else:
            print(f"[DEBUG] Completely failed to generate narrative arc for {life_skeleton.get('life_goal', 'Life Goal')}")
            return None
    
    processed_results = parallel_process(life_skeletons_with_uuid, process_single_item, "Stage3.1-Skeleton2Core")
    
    with jsonlines.open(output_file, 'w') as writer:
        for result in processed_results:
            if result is not None:
                writer.write(result)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate narrative arc persona information')
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
    
    input_file = os.getenv('STAGE2_3_FILE_PATH', 'data/stage2_3_profile2skeleton.jsonl')  # Previous step (stage2_3) output file
    output_file = os.getenv('STAGE3_1_FILE_PATH', 'data/stage3_1_narrative_arc.jsonl')  # Current step output file
    process_life_skeletons(input_file, output_file, regenerate)