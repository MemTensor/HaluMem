from datetime import datetime
import logging
import json
import os
import traceback
import jsonlines
from dotenv import load_dotenv
from typing import Dict, List
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


# Read configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    stage2_config = config['STAGE2']['substage3']
    stage3_config = config['STAGE3']['substage2']

# Read prompt from file
    try:
        with open('prompts/stage2_3_profile2skeleton_prompt.txt', 'r', encoding='utf-8') as f:
            profile2skeleton_prompt = f.read()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load stage2_3 prompt: {e}:{traceback.format_exc()}")

def sample_stage_info_for_dynamic_content(dynamic_content: Dict) -> Dict:
    """Sample stage information for dynamic content"""
    stage_info = {}
    
    for content_type, content_data in dynamic_content.items():
        # Check if it's new format (contains init and evolution_path)
        if isinstance(content_data, dict) and 'evolution_path' in content_data:
            # New format: get evolution information from evolution_path
            evolution_path = content_data['evolution_path']
            stage_count = len(evolution_path)
            
            # Create stage information for each evolution step
            for i, evolution in enumerate(evolution_path):
                stage_key = f"{content_type}_step_{i+1}"
                stage_info[stage_key] = {
                    "stage_count": 1,
                    "stage_type": evolution['direction'],  # Use evolution direction as stage type
                    "memory_points": [evolution['direction']]  # Evolution direction as memory point
                }
        else:
            # Old format: maintain original logic (compatibility)
            if content_type == "age":
                stage_info[content_type] = {
                    "stage_count": 1,
                    "stage_type": "modify",
                    "memory_points": ["current_age"]
                }
            elif content_type == "financial_status":
                stage_count_range = stage3_config['dynamic_stage_count_ranges'].get(content_type, [2, 4])
                stage_count = random.randint(stage_count_range[0], stage_count_range[1])
                stage_type = random.choice(["modify", "add", "delete"])
                memory_points = stage3_config['dynamic_memory_points'].get(content_type, ["income_level", "savings_status", "financial_pressure", "situation_reason"])
                if stage_type == "modify":
                    selected_points = random.sample(memory_points, random.randint(1, len(memory_points)))
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": selected_points
                    }
                else:
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": memory_points
                    }
            elif content_type == "health_status":
                stage_count_range = stage3_config['dynamic_stage_count_ranges'].get(content_type, [2, 4])
                stage_count = random.randint(stage_count_range[0], stage_count_range[1])
                stage_type = random.choice(["modify", "add", "delete"])
                memory_points = stage3_config['dynamic_memory_points'].get(content_type, ["physical_health", "mental_health", "physical_chronic_conditions", "mental_chronic_conditions", "situation_reason"])
                if stage_type == "modify":
                    selected_points = random.sample(memory_points, random.randint(1, len(memory_points)))
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": selected_points
                    }
                else:
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": memory_points
                    }
            elif content_type == "social_relationships":
                stage_count_range = stage3_config['dynamic_stage_count_ranges'].get(content_type, [2, 4])
                stage_count = random.randint(stage_count_range[0], stage_count_range[1])
                stage_type = random.choice(["modify", "add", "delete"])
                memory_points = stage3_config['dynamic_memory_points'].get(content_type, ["friends", "colleagues", "family", "romantic_relationship", "recent_changes"])
                if stage_type == "modify":
                    selected_points = random.sample(memory_points, random.randint(1, len(memory_points)))
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": selected_points
                    }
                else:
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": memory_points
                    }
            elif content_type == "family_life":
                stage_count = random.randint(2, 4)
                stage_type = random.choice(["modify", "add", "delete"])
                memory_points = ["parent_status", "partner_status", "child_status", "family_members", "family_description"]
                if stage_type == "modify":
                    selected_points = random.sample(memory_points, random.randint(1, len(memory_points)))
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": selected_points
                    }
                else:
                    stage_info[content_type] = {
                        "stage_count": stage_count,
                        "stage_type": stage_type,
                        "memory_points": memory_points
                    }
    
    return stage_info

def sample_stage_info_for_preferences(preferences: Dict) -> Dict:
    """Sample stage information for preferences content"""
    stage_info = {}
    
    for preference_type, preference_data in preferences.items():
        # Check if it's new format (contains init and evolution_path)
        if isinstance(preference_data, dict) and 'evolution_path' in preference_data:
            # New format: get evolution information from evolution_path
            evolution_path = preference_data['evolution_path']
            stage_count = len(evolution_path)
            
            # Create stage information for each evolution step
            for i, evolution in enumerate(evolution_path):
                stage_key = f"{preference_type}_step_{i+1}"
                stage_info[stage_key] = {
                    "stage_count": 1,
                    "stage_type": evolution['direction'],  # Use evolution direction as stage type
                    "memory_points": [evolution['direction']]  # Evolution direction as memory point
                }
        else:
            # Old format: maintain original logic (compatibility)
            stage_count = random.randint(1, 3)
            stage_type = random.choice(["modify", "add", "delete"])
            memory_points = ["preference_strength", "preference_frequency", "preference_scenario", "preference_reason", "specific_examples"]
            
            if stage_type == "modify":
                selected_points = random.sample(memory_points, random.randint(1, len(memory_points)))
                stage_info[preference_type] = {
                    "stage_count": stage_count,
                    "stage_type": stage_type,
                    "memory_points": selected_points
                }
            else:
                stage_info[preference_type] = {
                    "stage_count": stage_count,
                    "stage_type": stage_type,
                    "memory_points": memory_points
                }
    
    return stage_info

def extract_dynamic_steps(dynamic_stage_info: Dict) -> Dict[str, List[Dict]]:
    """Extract dynamic stage information into structure grouped by base_type and sorted by step in ascending order"""
    grouped: Dict[str, List[Dict]] = {}
    for content_type, stage_info in dynamic_stage_info.items():
        if "_step_" in content_type:
            base_type = content_type.split("_step_")[0]
            step_num = int(content_type.split("_step_")[1])
        else:
            base_type = content_type
            step_num = 1
        grouped.setdefault(base_type, []).append({
            "content_type": content_type,
            "base_type": base_type,
            "step": step_num,
            "stage_info": stage_info
        })
    # Sort
    for base_type in grouped:
        grouped[base_type].sort(key=lambda x: x["step"])  # Ascending order
    return grouped


def allocate_steps_to_events(grouped: Dict[str, List[Dict]], career_event_count: int) -> Dict[int, List[Dict]]:
    """Allocate steps to events:
    - Same type allocated to earlier events by step order
    - Single event can contain at most one step of same type
    - Balance load as much as possible while satisfying constraints
    """
    event_assignments: Dict[int, List[Dict]] = {i: [] for i in range(1, career_event_count + 1)}
    event_load: Dict[int, int] = {i: 0 for i in range(1, career_event_count + 1)}
    event_types: Dict[int, set] = {i: set() for i in range(1, career_event_count + 1)}

    for base_type in sorted(grouped.keys()):
        steps = grouped[base_type]
        last_event_idx = 0
        for step_item in steps:
            # First attempt: maintain time progression, choose events after last_event_idx that don't contain this base_type and have minimum load
            candidate_indices = [idx for idx in range(last_event_idx + 1, career_event_count + 1)
                                 if base_type not in event_types[idx]]
            if not candidate_indices:
                # Second attempt: relax time constraint (scan from beginning), still need to not contain this base_type
                candidate_indices = [idx for idx in range(1, career_event_count + 1)
                                     if base_type not in event_types[idx]]
            if not candidate_indices:
                # Extreme case (step count of this type greater than event count), cannot satisfy constraints.
                # To ensure complete allocation, choose event with minimum load and skip same-type constraint (try to avoid triggering).
                candidate_indices = list(range(1, career_event_count + 1))
            # Choose event with minimum load and minimum index
            candidate_indices.sort(key=lambda i: (event_load[i], i))
            chosen = candidate_indices[0]

            event_assignments[chosen].append(step_item)
            event_load[chosen] += 1
            event_types[chosen].add(base_type)
            last_event_idx = chosen

    return event_assignments


def assign_dynamic_content_to_events(dynamic_stage_info: Dict, career_event_count: int) -> Dict:
    """Assign dynamic content to career core events, call extracted steps: extract -> allocate"""
    grouped = extract_dynamic_steps(dynamic_stage_info)
    return allocate_steps_to_events(grouped, career_event_count)


def generate_career_event_template(career_event_count: int, event_assignments: Dict, persona: Dict) -> str:
    """Generate career core event template"""
    events_template = ""
    
    # Get birth date and current age
    birth_date = persona['fixed']['basic_info']['birth_date']
    current_age = persona['fixed']['age']['current_age']
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    current_day = current_date.day
    
    # Calculate event time range (starting from current time)
    start_year = current_year
    end_year = start_year + 10  # Next 10 years
    
    for event_id in range(1, career_event_count + 1):
        assigned_content = event_assignments.get(event_id, [])
        
        # Calculate event time (increment by order)
        event_year = start_year + (event_id - 1) * 2  # One event every 2 years
        
        # Ensure event start time is not earlier than current real date
        if event_year == current_year:
            # If current year, ensure month and day are not earlier than current date
            min_month = current_month
            min_day = current_day
        else:
            # If future year, can use any month and day
            min_month = 1
            min_day = 1
        
        # Generate event month and day, ensure not earlier than current date
        if event_year == current_year and min_month == 12:
            # If currently December, jump to next year
            event_year += 1
            event_month = random.randint(1, 12)
            event_day = random.randint(1, 28)
        else:
            event_month = random.randint(min_month, 12)
            if event_month == min_month:
                event_day = random.randint(min_day, 28)
            else:
                event_day = random.randint(1, 28)
        
        # Calculate age when event occurs
        event_age = current_age + (event_id - 1) * 2
        
        # Event duration (random 1-12 months)
        duration_months = random.randint(1, 12)
        end_month = event_month + duration_months
        end_year_adjust = 0
        if end_month > 12:
            end_month -= 12
            end_year_adjust = 1
        end_year_event = event_year + end_year_adjust
        
        events_template += f"""    {{
      "event_id": {event_id},
      "event_type": "career_event",
      "assigned_dynamic_content": {json.dumps(assigned_content, ensure_ascii=False, indent=4)},
      "event_name": "Career Event Name",
      "event_start_time": "{event_year:04d}-{event_month:02d}-{event_day:02d}",
      "event_end_time": "{end_year_event:04d}-{end_month:02d}-{event_day:02d}",
      "user_age": {event_age},
      "event_description": "Detailed description of career event background, process, and results",
      "event_result": "Specific results of career event"
    }}"""
        
        if event_id < career_event_count:
            events_template += ","
        events_template += "\n"
    
    return events_template

def generate_life_skeleton_json_template(career_event_count: int, event_assignments: Dict, persona: Dict) -> str:
    """Generate life skeleton JSON template"""
    events_template = generate_career_event_template(career_event_count, event_assignments, persona)
    
    # Build complete JSON template
    json_template = f"""
{{
  "life_goal": "Copy life goal once",
  "career_events": [
{events_template}  ],
  "life_events": [
    // Life events will be added in subsequent steps
  ]
}}"""
    
    return json_template


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_life_skeleton(persona: Dict, career_event_count: int, event_assignments: Dict, previous_cost: Dict = None) -> Dict:
    """Generate character's core event list"""
    # Build user input content, only include fixed and dynamic information
    persona_for_llm = {
        "fixed": persona['fixed'],
        "dynamic": persona['dynamic']
    }
    
    user_content = f"""
<Ultimate Life Goal>
{persona['fixed']['life_goal']}
</Ultimate Life Goal>

<Current Persona>
{json.dumps(persona_for_llm, ensure_ascii=False, indent=2)}
</Current Persona>

<Career Core Event Count>
{career_event_count}
</Career Core Event Count>

<Dynamic Content Assignment>
{json.dumps(event_assignments, ensure_ascii=False, indent=2)}
</Dynamic Content Assignment>

<JSON Output Template>
{generate_life_skeleton_json_template(career_event_count, event_assignments, persona)}
</JSON Output Template>
"""
    
    # Use prompt read from file
    system_prompt = profile2skeleton_prompt
    
    try:
        print("[DEBUG] Sending life skeleton generation request to LLM...")

        json_markers = [
            "Corrected life skeleton", "Generated life skeleton", "Final JSON", 
            "Complete JSON", "Correction result"
        ]

        raw_response, cost_info = llm_request(system_prompt, user_content, json_markers=json_markers)
        
        token_cost = calculate_cumulative_cost(previous_cost, cost_info)
        
        current_cost = cost_info
        cumulative_cost = token_cost.get('cumulative', {})
        print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
              f"Output: {current_cost.get('output_tokens', 'N/A')}, "
              f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
        print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
              f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        try:
            life_skeleton = _extract_json_from_content(raw_response, json_markers)
            life_skeleton['persona_update_dict'] = {}
            print(f"[DEBUG] Successfully parsed JSON")

            return {
                "life_skeleton": life_skeleton,
                "raw_response": raw_response,
                "token_cost": token_cost
            }
        except ValueError as e:
            print(f"[DEBUG] JSON parsing error: {e}:{traceback.format_exc()}")
            raise
        
    except Exception as e:
        print(f"[DEBUG] Error calling large language model to generate event details: {e}:{traceback.format_exc()}")
        raise

def update_persona(persona: Dict, updated_persona: Dict) -> Dict:
    """Recursively update nested persona structure"""
    for key, value in updated_persona.items():
        if isinstance(value, dict) and key in persona:
            # If value is dict and key exists in persona, recursively update
            update_persona(persona[key], value)
        else:
            # Otherwise update directly
            persona[key] = value
    return persona

def read_persona_from_file(file_path: str) -> List[Dict]:
    """Read persona information and corresponding UUID from file"""
    personas_with_uuid = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            if 'persona' in obj and 'uuid' in obj:
                personas_with_uuid.append({
                    'persona': obj['persona'],
                    'uuid': obj['uuid'],
                    'metadata': obj.get('metadata', {})  # 读取 metadata 以获取成本信息
                })
    return personas_with_uuid

def process_personas(file_path: str, output_file: str, regenerate: bool = True):
    """Process all personas, generate core event list and save"""
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
            raise RuntimeError(f"Error checking output file: {e}:{traceback.format_exc()}")
    
    # Read data from input file
    try:
        data = read_persona_from_file(file_path)
        if not data:
            raise ValueError(f"Input file {file_path} does not contain persona entries")
        print(f"[DEBUG] Successfully read {file_path}, total {len(data)} entries")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}:{traceback.format_exc()}")
    
    def process_single_item(item):
        persona = item['persona']
        persona_uuid = item['uuid']
        
        print(f"[DEBUG] Processing persona: {persona['fixed']['basic_info']['name']} (UUID: {persona_uuid})")
        
        # 1. Use stage count passed from first substage
        career_event_count = persona.get('career_event_count', None)
        if career_event_count is None:
            print(f"[ERROR] Stage count passed from substage not found, skipping processing")
            return None
        else:
            print(f"[DEBUG] Using passed stage count: {career_event_count}")
        
        # 2. Sample stage information for each dynamic content
        dynamic_stage_info = sample_stage_info_for_dynamic_content(persona.get('dynamic', {}))
        print(f"[DEBUG] Dynamic stage info: {dynamic_stage_info}")
        
        # 3. Sample stage information for each preference content
        preference_stage_info = sample_stage_info_for_preferences(persona.get('preferences', {}))
        print(f"[DEBUG] Preference stage info: {preference_stage_info}")
        if not preference_stage_info:
            raise ValueError(f"No preference stage info generated for persona {persona_uuid}")
        
        # 4. Assign dynamic content to career events
        event_assignments = assign_dynamic_content_to_events(dynamic_stage_info, career_event_count)
        print(f"[DEBUG] Event assignments: {event_assignments}")
        
        # 5. Retrieve cost information from previous stages and call the generate_life_skeleton function
        previous_cost = item.get('token_cost')
        
        result = generate_life_skeleton(persona, career_event_count, event_assignments, previous_cost)
        
        if result:
            # Build new output format，include token_cost
            output_data = {
                "uuid": persona_uuid,
                "life_skeleton": result.get('life_skeleton', {}),
                "profile": persona,
                "metadata": {
                    "persona_profile": persona,
                    "dynamic_stage_info": dynamic_stage_info,
                    "preference_stage_info": preference_stage_info,
                    "career_event_count": career_event_count,
                    "event_assignments": event_assignments,
                    "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "token_cost": result.get('token_cost')
            }
            if result.get('life_skeleton'):
                print(f"[DEBUG] Successfully generated life skeleton for {persona['fixed']['basic_info']['name']}")
            else:
                print(f"[DEBUG] Failed to generate life skeleton, but saved original response for {persona['fixed']['basic_info']['name']}")
            return output_data
        else:
            print(f"[DEBUG] Completely failed to generate life skeleton for {persona['fixed']['basic_info']['name']}")
            return None
    
    processed_results = parallel_process(data, process_single_item, "Stage2.3-Profile2Skeleton")
    
    with jsonlines.open(output_file, 'w') as writer:
        for result in processed_results:
            if result is not None:
                writer.write(result)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate life skeleton persona information')
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
    
    input_file = os.getenv('STAGE2_2_FILE_PATH', 'data/stage2_2_extend_preference.jsonl')
    output_file = os.getenv('STAGE2_3_FILE_PATH', 'data/stage2_3_profile2skeleton.jsonl')
    process_personas(input_file, output_file, regenerate)