import json
import random
import os
import jsonlines
import traceback
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any

from llm_request import calculate_cumulative_cost
from parallel_utils import parallel_process

# Load environment variables
load_dotenv()

# Read configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE2']['substage2']


def sample_evolution_for_preference_content(preference_content: Dict) -> Dict:
    """Sample evolution information for preference content"""
    evolution_config = config['preference_evolution']
    evolution_count_range = evolution_config['evolution_count_range']
    evolution_directions = evolution_config['evolution_directions']
    
    extended_preferences = {}
    
    for preference_type, preference_data in preference_content.items():
        # Sample evolution count
        evolution_count = random.randint(
            evolution_count_range[0], 
            evolution_count_range[1]
        )
        
        # Sample evolution directions (unified three directions)
        evolution_path = []
        
        # Track current memory points count for protecting delete operations
        current_memory_points_count = len(preference_data.get('memory_points', []))
        
        for i in range(evolution_count):
            # If currently only one memory point remains, cannot delete, need to resample
            max_retries = 10000
            retry_count = 0
            direction = None
            
            while retry_count < max_retries:
                sampled_direction = random.choices(
                    list(evolution_directions.keys()),
                    weights=list(evolution_directions.values())
                )[0]
                
                # If delete is sampled and currently only one memory point remains, resample
                if sampled_direction == "Delete" and current_memory_points_count <= 1:
                    retry_count += 1
                    continue
                else:
                    direction = sampled_direction
                    break
            
            if direction is None:
                raise ValueError(
                    f"Unable to sample valid evolution direction for preference '{preference_type}' at step {i + 1}"
                )
            
            # Update current memory points count (simplified handling: delete -1, modify +0, add +1)
            if direction == "Delete":
                current_memory_points_count = max(0, current_memory_points_count - 1)
            elif direction == "Add":
                current_memory_points_count += 1
            
            evolution_path.append({
                "step": i + 1,
                "direction": direction
            })
        
        # Build new format
        extended_preferences[preference_type] = {
            "init": preference_data,  # Initial state
            "evolution_path": evolution_path  # Evolution path
        }
    
    return extended_preferences


def extend_preference_content(persona_data: Dict) -> Dict:
    """Substage 2: Extend preference content"""
    print("[DEBUG] Substage 2: Extending preference content...")
    
    # Get original preference content
    original_preferences = persona_data['profile'].get('preferences', {})
    
    # Sample evolution information and adjust format
    extended_preferences = sample_evolution_for_preference_content(original_preferences)
    
    # Update persona data
    result = persona_data.copy()
    persona = result.get('persona') or result.get('profile')
    if persona is None:
        raise ValueError("Persona data missing 'profile'/'persona' structure")
    persona['preferences'] = extended_preferences
    if 'career_event_count' in persona_data.get('profile', {}):
        persona['career_event_count'] = persona_data['profile']['career_event_count']
    result['persona'] = persona
    result.pop('profile', None)
    
    return result


def process_single_persona(persona_data: Dict) -> Dict:
    """Process single persona data, extend preference content"""
    print(f"[DEBUG] Processing persona: {persona_data.get('uuid', 'unknown')}")
    
    previous_cost = persona_data.get('token_cost')
    
    # Substage 2: Extend preference content
    print("[DEBUG] Substage 2: Extending preference content...")
    extended_persona = extend_preference_content(persona_data)
    print(f"[DEBUG] Preference content extension completed")
    
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
    
    cumulative_cost = token_cost.get('cumulative', {})
    print(f"[DEBUG] No LLM calls in current stage")
    print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
          f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
    
    # Update persona data
    result = extended_persona.copy()
    
    cleaned_metadata = {
        'persona_seed': persona_data['metadata']['persona_seed'],
        'extend_preference_generate_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    result['metadata'] = cleaned_metadata
    
    result['token_cost'] = token_cost
    
    print("[DEBUG] Processing completed")
    return result


def process_all_personas(regenerate: bool = True):
    """Process all personas, extend preference content for each"""
    input_file = os.getenv('STAGE2_1_FILE_PATH', 'data/stage2_1_extend_dynamic.jsonl')
    output_file = os.getenv('STAGE2_2_FILE_PATH', 'data/stage2_2_extend_preference.jsonl')
    
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
        processed_personas = parallel_process(all_personas, process_single_persona, "Stage2.2-ExtendPreference")
        
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
    parser = argparse.ArgumentParser(description='Extend preference content persona information')
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
