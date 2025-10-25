import json
import random
import os
import traceback
import jsonlines
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Set, Tuple

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


# Load configuration file (keep complete configuration for cross-substage data access, e.g., name database)
with open('config.json', 'r', encoding='utf-8') as f:
    _full_config = json.load(f)
    config = _full_config['STAGE1']['substage2']
    _names_config = _full_config['STAGE1']['substage1']['names']

# Read prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")




# Load second step prompt
stage1_2_dynamic_prompt = load_prompt(config['prompts']['stage1_2_dynamic'])
if not stage1_2_dynamic_prompt:
    print("[ERROR] Unable to load stage1_2_dynamic prompt file")
    exit(1)
print(f"[DEBUG] Successfully loaded stage1_2_dynamic prompt, length: {len(stage1_2_dynamic_prompt)}")


def generate_dynamic_persona(fixed_persona: Dict) -> Dict:
    """Step 2: Generate initial values for dynamic part"""
    print("[DEBUG] Step 2: Generating dynamic part...")
    
    # Use new configuration structure to generate career status information
    career_status_config = config['career_status']
    
    # Select employment status based on probability distribution
    employment_status = random.choices(
        list(career_status_config['employment_status'].keys()),
        weights=list(career_status_config['employment_status'].values())
    )[0]
    
    # Select monthly income and savings amount based on probability distribution, and sample specific values from ranges
    def _sample_value_from_range_str(range_str: str) -> int:
        try:
            s = str(range_str).replace(' ', '')

            if s.endswith('+'):
                low = int(float(s[:-1]))
                high = int(low * 2)
                value = random.randint(low, high)

            elif '-' in s:
                parts = s.split('-')
                if len(parts) == 2:
                    low = int(float(parts[0]))
                    high = int(float(parts[1]))
                    if high < low:
                        low, high = high, low
                    value = random.randint(low, high)
                else:
                    raise ValueError(f"Invalid range format: {s}")

            else:
                value = int(float(s))
            
        except Exception as e:
            raise ValueError(f"Invalid range string '{range_str}': {e}:{traceback.format_exc()}")

        # Only keep first three significant digits, fill the rest with 0s for better readability
        s = str(abs(value))
        if len(s) <= 3:
            norm = value
        else:
            norm = int(s[:3] + ('0' * (len(s) - 3)))
            if value < 0:
                norm = -norm
        return norm

    monthly_income_key = random.choices(
        list(career_status_config['monthly_income'].keys()),
        weights=list(career_status_config['monthly_income'].values())
    )[0]
    monthly_income = _sample_value_from_range_str(monthly_income_key)

    savings_amount_key = random.choices(
        list(career_status_config['savings_amount'].keys()),
        weights=list(career_status_config['savings_amount'].values())
    )[0]
    savings_amount = _sample_value_from_range_str(savings_amount_key)
    
    # Generate corresponding career information based on employment status
    career_status = {
        "employment_status": employment_status,
        "industry": "",
        "company_name": "",
        "job_title": "",
        "monthly_income": monthly_income,
        "savings_amount": savings_amount,
        "career_description": "Career status description to be determined"
    }
    
    if employment_status == "employed":
        # Employed: generate complete career information
        company_type = random.choices(
            list(career_status_config['company_types'].keys()),
            weights=list(career_status_config['company_types'].values())
        )[0]
        
        job_title = random.choices(
            list(career_status_config['job_titles'].keys()),
            weights=list(career_status_config['job_titles'].values())
        )[0]
        
        industry = random.choices(
            list(career_status_config['industries'].keys()),
            weights=list(career_status_config['industries'].values())
        )[0]
        
        # Get company name from configuration
        company_name = random.choice(career_status_config['company_names'][company_type])
        
        career_status.update({
            "company_name": company_name,
            "job_title": job_title,
            "industry": industry
        })
        
    elif employment_status == "entrepreneur":
        # Entrepreneur: generate entrepreneurship-related information
        industry = random.choices(
            list(career_status_config['industries'].keys()),
            weights=list(career_status_config['industries'].values())
        )[0]
        
        career_status.update({
            "company_name": "Self-founded Company",
            "job_title": "Founder/CEO",
            "industry": industry
        })
        
    else:
        # Unemployed: keep default values
        pass
    
    # Generate health status using probability distribution
    physical_status_levels = config['health']['physical']['overall_status']
    mental_status_levels = config['health']['mental']['overall_status']
    
    physical_health = random.choices(
        list(physical_status_levels.keys()),
        weights=list(physical_status_levels.values())
    )[0]
    
    mental_health = random.choices(
        list(mental_status_levels.keys()),
        weights=list(mental_status_levels.values())
    )[0]
    
    # Generate social relationships list (changed to dictionary structure with other person's name as key)
    relationship_count_range = config['social_relationships']['relationship_count_range']
    relationship_types = config['social_relationships']['relationship_types']
    
    # Randomly select social relationship count
    relationship_count = random.randint(relationship_count_range[0], relationship_count_range[1])
    
    # Select social relationship types based on probability distribution
    selected_relationship_types = random.choices(
        list(relationship_types.keys()),
        weights=list(relationship_types.values()),
        k=relationship_count
    )
    
    # Helper function to generate names
    def sample_other_person_name() -> str:
        family_name = random.choice(_names_config.get('family_names', ['Zhang']))
        # Random gender for naming
        if random.random() < 0.5:
            given_name = random.choice(_names_config.get('male_names', ['Wei']))
        else:
            given_name = random.choice(_names_config.get('female_names', ['Min']))
        return family_name + given_name

    # Generate social relationships (dictionary), key is other person's name, value is original single structure
    social_relationships: Dict[str, Dict[str, str]] = {}
    used_names = set()
    for rel_type in selected_relationship_types:
        # Ensure name uniqueness, avoid key conflicts
        for _ in range(5):
            candidate = sample_other_person_name()
            if candidate not in used_names:
                used_names.add(candidate)
                break
        else:
            # If multiple conflicts, add random suffix
            candidate = sample_other_person_name() + str(random.randint(1, 99))
            used_names.add(candidate)

        social_relationships[candidate] = {
            "relationship_type": rel_type,
            "description": f"{rel_type} relationship description to be determined"
        }
    

    
    dynamic_persona = {
        "career_status": career_status,
        "health_status": {
            "physical_health": physical_health,
            "physical_chronic_conditions": "",
            "mental_health": mental_health,
            "mental_chronic_conditions": "",
            "situation_reason": "Reasons for such health status"
        },
        "social_relationships": social_relationships
    }
    
    return dynamic_persona


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def validate_and_correct_dynamic_persona(base_persona: str, fixed_persona: Dict, dynamic_persona: Dict, previous_cost: Dict = None) -> tuple:
    """Use large language model to validate and correct dynamic part persona information"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("[DEBUG] API key not set, returning original dynamic_persona")
        return dynamic_persona, None, {"token_cost": None}
    
    # Build user input content
    user_content = f"""
Original persona seed: {base_persona}

Fixed part information:
{json.dumps(fixed_persona, ensure_ascii=False, indent=2)}

Currently generated dynamic part persona information:
{json.dumps(dynamic_persona, ensure_ascii=False, indent=2)}

Please analyze and correct the above dynamic part persona information, generating reasonable initial descriptions for each dynamic item (point-by-point description).
"""
    
    # Use specialized prompt for step 2
    system_prompt = stage1_2_dynamic_prompt
    
    try:
        print("[DEBUG] Sending dynamic part correction request to LLM...")

        user_content = f"\nOriginal persona seed: {base_persona}\n\n" + \
                       "Currently generated persona information:\n" + \
                       f"{json.dumps(dynamic_persona, ensure_ascii=False, indent=2)}\n\n" + \
                       "Please analyze and correct the above persona information, " + \
                       "and present the final result only as valid JSON. The JSON must be wrapped " + \
                       "inside a Markdown code block: ```json```."
        
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_dynamic_persona, cost_info = llm_request(
            system_prompt, 
            user_content, 
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)
        
        print(f"[DEBUG] Successfully processed dynamic part with LLM caller")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        return corrected_dynamic_persona['Corrected Dynamic Part'], cost_info
        
    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}:{traceback.format_exc()}")
        raise 


def process_single_persona(persona_data: Dict) -> Dict:
    """Process single persona data, add dynamic part"""
    print(f"[DEBUG] Processing persona: {persona_data.get('uuid', 'unknown')}")
    
    # Get persona seed and fixed part
    persona_seed = persona_data['metadata']['persona_seed']
    fixed_persona = persona_data['profile']['fixed']
    
    previous_cost = persona_data.get('token_cost')
    
    # Step 2: Generate dynamic part
    print("[DEBUG] Step 2: Generating dynamic part...")
    initial_dynamic_persona = generate_dynamic_persona(fixed_persona)
    corrected_dynamic_persona, cost_info = validate_and_correct_dynamic_persona(
        persona_seed, 
        fixed_persona, 
        initial_dynamic_persona,
        previous_cost
    )
    print(f"[DEBUG] Dynamic part generation completed")
    
    token_cost = cost_info
    
    # Update persona data, add dynamic part under profile
    result = persona_data.copy()
    result['profile']['dynamic'] = corrected_dynamic_persona
    
    cleaned_metadata = {
        'persona_seed': persona_seed,
        'dynamic_generate_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    result['metadata'] = cleaned_metadata
    
    if token_cost:
        result['token_cost'] = token_cost
    
    print("[DEBUG] Processing completed")
    return result


def process_all_personas(regenerate: bool = True):
    """Process all personas, add dynamic part for each"""
    input_file = os.getenv('STAGE1_1_FILE_PATH', 'data/stage1_1_fixed.jsonl')
    output_file = os.getenv('STAGE1_2_FILE_PATH', 'data/stage1_2_dynamic.jsonl')
    
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
        processed_personas = parallel_process(all_personas, process_single_persona, "Stage1.2-Dynamic")

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
    parser = argparse.ArgumentParser(description='Generate dynamic part persona information')
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