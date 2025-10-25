import json
import random
import os
import traceback
import jsonlines
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

from llm_request import calculate_cumulative_cost
from parallel_utils import parallel_process

# Load environment variables
load_dotenv()

# Read configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE2']['substage1']


def sample_evolution_for_dynamic_content(dynamic_content: Dict, career_event_count: int) -> Dict:
    """Sample evolution information for dynamic content, generate evolution based on sampled stage count"""
    evolution_config = config['dynamic_evolution']
    evolution_directions = evolution_config['evolution_directions']
    
    extended_dynamic = {}
    
    for content_type, content_data in dynamic_content.items():
        if content_type in evolution_directions:
            # Use sampled stage count as evolution count
            evolution_count = career_event_count
            available_directions = evolution_directions[content_type]
            evolution_path = []
            
            # Special handling for health status: add "Maintain Status" option
            if content_type == "health_status":
                # Use evolution directions from configuration (includes "Maintain Status" option)
                available_directions = evolution_directions[content_type]
                
                for i in range(evolution_count):
                    direction = random.choices(
                        list(available_directions.keys()),
                        weights=list(available_directions.values())
                    )[0]
                    
                    # If "Maintain Status" is sampled, don't actually write this item
                    if direction != "Maintain Status":
                        evolution_path.append({
                            "step": i + 1,
                            "direction": direction
                        })
                    # If "Maintain Status" is sampled, skip this step, don't add to evolution_path
            
            # Other content types: all dynamic content evolves at each stage
            else:
                for i in range(evolution_count):
                    direction = random.choices(
                        list(available_directions.keys()),
                        weights=list(available_directions.values())
                    )[0]
                    
                    # If "Maintain Status" is sampled, don't actually write this item
                    if direction != "Maintain Status":
                        evolution_path.append({
                            "step": i + 1,
                            "direction": direction
                        })
                    # If "Maintain Status" is sampled, skip this step, don't add to evolution_path
            
            # Build new format
            extended_dynamic[content_type] = {
                "init": content_data,  # Initial state
                "evolution_path": evolution_path  # Evolution path
            }
        else:
            # If not in evolution configuration, keep original format
            extended_dynamic[content_type] = content_data
    
    return extended_dynamic


def extend_dynamic_content(persona_data: Dict) -> Dict:
    """Substage 1: Extend dynamic content"""
    print("[DEBUG] Substage 1: Extending dynamic content...")
    
    # Sample skeleton stage count in first substage
    career_event_count = random.randint(config['core_event_count_range'][0], config['core_event_count_range'][1])
    print(f"[DEBUG] Sampled skeleton stage count: {career_event_count}")
    
    # Get original dynamic content
    original_dynamic = persona_data['profile'].get('dynamic', {})
    
    # Generate evolution information based on sampled stage count and adjust format
    extended_dynamic = sample_evolution_for_dynamic_content(original_dynamic, career_event_count)
    
    # Update persona data
    result = persona_data.copy()
    result['profile']['dynamic'] = extended_dynamic
    
    # Record sampled stage count to top level
    result['profile']['career_event_count'] = career_event_count
    
    return result


def process_single_persona(persona_data: Dict) -> Dict:
    """Process single persona data, extend dynamic content"""
    print(f"[DEBUG] Processing persona: {persona_data.get('uuid', 'unknown')}")
    
    previous_cost = persona_data.get('token_cost')
    
    # Substage 1: Extend dynamic content
    print("[DEBUG] Substage 1: Extending dynamic content...")
    extended_persona = extend_dynamic_content(persona_data)
    print(f"[DEBUG] Dynamic content extension completed")
    
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
        'extend_dynamic_generate_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    result['metadata'] = cleaned_metadata
    
    result['token_cost'] = token_cost
    
    print("[DEBUG] Processing completed")
    return result


def process_all_personas(regenerate: bool = True):
    """Process all personas, extend dynamic content for each"""
    input_file = os.getenv('STAGE1_3_FILE_PATH', 'data/stage1_3_preferences.jsonl')
    output_file = os.getenv('STAGE2_1_FILE_PATH', 'data/stage2_1_extend_dynamic.jsonl')
    
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
        
        # Process each persona
        processed_personas = parallel_process(all_personas, process_single_persona, "Stage2.1-ExtendDynamic")
        
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
    parser = argparse.ArgumentParser(description='Extend dynamic content persona information')
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
