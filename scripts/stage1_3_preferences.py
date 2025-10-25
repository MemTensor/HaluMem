import json
import random
import os
import traceback
import jsonlines
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Set, Tuple

# 导入统一的 LLM 调用器
# from llm_caller import LLMCaller, validate_and_correct_with_llm
from llm_request import llm_request, calculate_cumulative_cost
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
    config = json.load(f)
    config = config['STAGE1']['substage3']

# Read prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[DEBUG] Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")
        return ""

# Load third step prompt
stage1_3_preferences_prompt = load_prompt(config['prompts']['stage1_3_preferences'])


def generate_preferences_persona() -> Dict:
    """Step 3: Generate initial values for preferences part"""
    print("[DEBUG] Step 3: Generating preferences part...")
    
    # Randomly select preference types from configuration
    preference_types = list(config['preferences']['preference_types'].keys())
    selected_preferences = random.sample(
        preference_types,
        random.randint(
            config['preferences']['preference_types_count_range'][0],
            config['preferences']['preference_types_count_range'][1]
        )
    )
    
    preferences = {}
    
    for pref_type in selected_preferences:
        # Get configuration for this preference type
        pref_config = config['preferences']['preference_types'][pref_type]
        
        # Use unified memory point count range
        memory_count = random.randint(
            config['preferences']['preference_mempoints_count_range'][0],
            config['preferences']['preference_mempoints_count_range'][1]
        )
        
        # Generate memory points list
        memory_points = []
        for i in range(memory_count):
            # Randomly select like or dislike
            binary_type = random.choice(list(pref_config['binary_types'].keys()))
            
            memory_point = {
                "type": binary_type,
                "type_description": pref_config['binary_types'][binary_type],
                "specific_item": f"Specific item {i+1} to be determined",
                "reason": f"Reason {i+1} to be determined"
            }
            memory_points.append(memory_point)
        
        # Build complete structure for this preference type
        preferences[pref_type] = {
            "memory_points": memory_points
        }
    
    return preferences


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def validate_and_correct_preferences_persona(base_persona: str, fixed_persona: Dict, preferences_persona: Dict, previous_cost: Dict = None) -> tuple:
    """Use large language model to validate and correct preferences part persona information"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("[DEBUG] API key not set, returning original preferences_persona")
        return preferences_persona, None, {"token_cost": None}
    
    # Build user input content
    user_content = f"""
Original persona seed: {base_persona}

Fixed part information:
{json.dumps(fixed_persona, ensure_ascii=False, indent=2)}

Currently generated preferences part persona information:
{json.dumps(preferences_persona, ensure_ascii=False, indent=2)}

Please analyze and correct the above preferences part persona information, ensuring each preference is detailed and meets the requirements of the preference description.
"""
    
    # Use specialized prompt for step 3
    system_prompt = stage1_3_preferences_prompt
    
    try:
        print("[DEBUG] Sending preferences part correction request to LLM...")
        
        user_content = f"\nOriginal persona seed: {base_persona}\n\n" + \
                       "Currently generated persona information:\n" + \
                       f"{json.dumps(preferences_persona, ensure_ascii=False, indent=2)}\n\n" + \
                       "Please analyze and correct the above persona information, " + \
                       "and present the final result only as valid JSON. The JSON must be wrapped " + \
                       "inside a Markdown code block: ```json```."
        
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_preferences_persona, cost_info = llm_request(
            system_prompt, 
            user_content, 
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)
        
        print(f"[DEBUG] Successfully processed preferences part with LLM caller")
        
        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        return corrected_preferences_persona['Corrected Preferences Part'], cost_info
        
    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}:{traceback.format_exc()}")
        raise


def process_single_persona(persona_data: Dict) -> Dict:
    """Process single persona data, add preferences part"""
    print(f"[DEBUG] Processing persona: {persona_data.get('uuid', 'unknown')}")
    
    # Get persona seed and fixed part
    persona_seed = persona_data['metadata']['persona_seed']
    fixed_persona = persona_data['profile']['fixed']
    
    previous_cost = persona_data.get('token_cost')
    
    # Step 3: Generate preferences part
    print("[DEBUG] Step 3: Generating preferences part...")
    initial_preferences_persona = generate_preferences_persona()
    corrected_preferences_persona, cost_info = validate_and_correct_preferences_persona(
        persona_seed, 
        fixed_persona, 
        initial_preferences_persona,
        previous_cost
    )
    print(f"[DEBUG] Preferences part generation completed")
    
    token_cost = cost_info
    
    # Update persona data, add preferences part under profile
    result = persona_data.copy()
    result['profile']['preferences'] = corrected_preferences_persona
    
    cleaned_metadata = {
        'persona_seed': persona_seed,
        'preferences_generate_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    result['metadata'] = cleaned_metadata
    
    if token_cost:
        result['token_cost'] = token_cost
    
    print("[DEBUG] Processing completed")
    return result


def process_all_personas(regenerate: bool = True):
    """Process all personas, add preferences part for each"""
    input_file = os.getenv('STAGE1_2_FILE_PATH', 'data/stage1_2_dynamic.jsonl')
    output_file = os.getenv('STAGE1_3_FILE_PATH', 'data/stage1_3_preferences.jsonl')
    
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
    
    try:
        # Read all persona data
        all_personas = []
        with jsonlines.open(input_file) as reader:
            for item in reader:
                all_personas.append(item)
        
        print(f"[DEBUG] Read {len(all_personas)} personas")
        
        # Process each persona - 并行版本
        processed_personas = parallel_process(all_personas, process_single_persona, "Stage1.3-Preferences")
        
        # Save all processed personas
        if processed_personas:
            with jsonlines.open(output_file, 'w') as writer:
                for persona in processed_personas:
                    writer.write(persona)
            print(f"[DEBUG] Successfully processed and saved {len(processed_personas)} personas")
        else:
            print("[DEBUG] No successfully processed personas")
        
        return True
        
    except Exception as e:
        print(f"Error processing personas: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate preferences part persona information')
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
    process_all_personas(regenerate)

