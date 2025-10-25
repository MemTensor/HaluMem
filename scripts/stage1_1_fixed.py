import json
import random
import os
import traceback
import jsonlines
import hashlib
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

from llm_request import llm_request, calculate_cumulative_cost
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
    config = config['STAGE1']['substage1']

# Read prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")

# Load first step prompt
stage1_1_fixed_prompt = load_prompt(config['prompts']['stage1_1_fixed'])


def generate_name(gender: str) -> str:
    """Generate English name based on gender"""
    family = random.choice(config['names']['family_names'])
    if gender == config['names']['genders'][0]:  # Male
        return family + random.choice(config['names']['male_names'])
    else:
        return family + random.choice(config['names']['female_names'])


def generate_location() -> str:
    """Generate location"""
    return random.choice(config['locations']['major_cities'])


def generate_birth_date() -> str:
    """Generate birth date, return string in YYYY-MM-DD format"""
    # Use birth date range settings from configuration file
    year_range = config['birth_date_range']['year_range']
    month_range = config['birth_date_range']['month_range']
    day_range = config['birth_date_range']['day_range']
    
    birth_year = random.randint(year_range[0], year_range[1])
    birth_month = random.randint(month_range[0], month_range[1])
    birth_day = random.randint(day_range[0], day_range[1])
    
    # Format as YYYY-MM-DD string
    return f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"


def generate_birth_date_from_age_diff(base_birth_date: str, age_diff: int) -> str:
    """Generate new birth date based on base birth date and age difference"""
    try:
        # Parse base birth date
        base_date = datetime.strptime(base_birth_date, "%Y-%m-%d")
        
        # Calculate target birth date (negative age difference means earlier birth)
        target_date = base_date.replace(year=base_date.year - age_diff)
        
        # Format as YYYY-MM-DD
        return target_date.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Error generating birth date from {base_birth_date} with age diff {age_diff}: {e}:{traceback.format_exc()}")


def generate_family_life(age: int, birth_date: str) -> Dict:
    """Generate family life information"""
    family_life_config = config['family_life']
    age_based_states = family_life_config['age_based_family_states']
    age_rules = family_life_config['family_member_age_rules']
    typical_ages = family_life_config['typical_ages']
    
    # Determine family status based on age
    # Find nearest typical age
    def find_nearest_typical_ages(target_age, typical_ages):
        """Find typical ages closest to target age"""
        typical_ages = sorted(typical_ages)
        
        # If target age is less than or equal to minimum typical age
        if target_age <= typical_ages[0]:
            return typical_ages[0], typical_ages[0]
        
        # If target age is greater than or equal to maximum typical age
        if target_age >= typical_ages[-1]:
            return typical_ages[-1], typical_ages[-1]
        
        # Find interval containing target age
        for i in range(len(typical_ages) - 1):
            if typical_ages[i] <= target_age <= typical_ages[i + 1]:
                return typical_ages[i], typical_ages[i + 1]
        
        return typical_ages[0], typical_ages[0]  # Default return
    
    lower_age, upper_age = find_nearest_typical_ages(age, typical_ages)
    
    # If there's only one typical age, use it directly
    if lower_age == upper_age:
        family_states = age_based_states[str(lower_age)]
    else:
        # Calculate interpolation weight
        weight = (age - lower_age) / (upper_age - lower_age)
        
        # Get states for two typical ages
        lower_states = age_based_states[str(lower_age)]
        upper_states = age_based_states[str(upper_age)]
        
        # Interpolate probabilities
        def interpolate_probabilities(lower_probs, upper_probs, weight):
            """Interpolate probabilities"""
            interpolated = {}
            for key in lower_probs.keys():
                interpolated[key] = lower_probs[key] * (1 - weight) + upper_probs[key] * weight
            return interpolated
        
        # Interpolate probabilities for various states
        parent_status_probs = interpolate_probabilities(
            lower_states['parent_status'], 
            upper_states['parent_status'], 
            weight
        )
        partner_status_probs = interpolate_probabilities(
            lower_states['partner_status'], 
            upper_states['partner_status'], 
            weight
        )
        child_status_probs = interpolate_probabilities(
            lower_states['child_status'], 
            upper_states['child_status'], 
            weight
        )
        
        family_states = {
            'parent_status': parent_status_probs,
            'partner_status': partner_status_probs,
            'child_status': child_status_probs
        }
    
    # Select family status based on probability distribution
    parent_status = random.choices(
        list(family_states['parent_status'].keys()),
        weights=list(family_states['parent_status'].values())
    )[0]
    
    partner_status = random.choices(
        list(family_states['partner_status'].keys()),
        weights=list(family_states['partner_status'].values())
    )[0]
    
    child_status = random.choices(
        list(family_states['child_status'].keys()),
        weights=list(family_states['child_status'].values())
    )[0]
    
    # Generate family members (changed to parent_members / partner / child_members separately)
    parent_members = []
    partner = None
    child_members = []
    
    # Add parent information
    if parent_status != "both_deceased":
        parent_age_diff = random.randint(age_rules['parent_age_range'][0], age_rules['parent_age_range'][1])
        if parent_status == "both_alive":
            parent_members.append({
                "member_type": "Father",
                "birth_date": generate_birth_date_from_age_diff(birth_date, parent_age_diff),
                "description": "Father description to be determined"
            })
            parent_members.append({
                "member_type": "Mother", 
                "birth_date": generate_birth_date_from_age_diff(birth_date, parent_age_diff),
                "description": "Mother description to be determined"
            })
        else:  # one_deceased
            if random.choice([True, False]):
                parent_members.append({
                    "member_type": "Father",
                    "birth_date": generate_birth_date_from_age_diff(birth_date, parent_age_diff),
                    "description": "Father description to be determined"
                })
            else:
                parent_members.append({
                    "member_type": "Mother",
                    "birth_date": generate_birth_date_from_age_diff(birth_date, parent_age_diff),
                    "description": "Mother description to be determined"
                })
    
    # Add partner information
    if partner_status in ["dating", "married", "divorced"]:
        partner_age_diff = random.randint(age_rules['partner_age_range'][0], age_rules['partner_age_range'][1])
        partner = {
            "member_type": "Partner",
            "birth_date": generate_birth_date_from_age_diff(birth_date, partner_age_diff),
            "description": "Partner description to be determined"
        }
    
    # Add child information
    if child_status != "no_children":
        child_count = 1 if child_status == "one_child" else (2 if child_status == "two_children" else 3)
        for i in range(child_count):
            child_age = random.randint(0, min(age_rules['child_age_range'][1], age - 18))
            child_members.append({
                "member_type": f"Child{i+1}",
                "birth_date": generate_birth_date_from_age_diff(birth_date, -child_age),
                "description": f"Child{i+1} description to be determined"
            })
    
    family_life = {
        "parent_status": parent_status,
        "partner_status": partner_status,
        "child_status": child_status,
        "parent_members": parent_members,
        "partner": partner,
        "child_members": child_members,
        "family_description": "Family life description to be determined"
    }
    
    return family_life


def generate_fixed_persona(base_persona: str) -> Dict:
    """Step 1: Generate fixed part persona information"""
    print("[DEBUG] Step 1: Generating fixed part...")
    
    # Randomly select gender
    gender = random.choice(config['names']['genders'])
    
    # Generate birth date
    birth_date = generate_birth_date()
    
    # Calculate age
    birth_datetime = datetime.strptime(birth_date, "%Y-%m-%d")
    current_datetime = datetime.now()
    
    age = current_datetime.year - birth_datetime.year
    if current_datetime.month < birth_datetime.month or (current_datetime.month == birth_datetime.month and current_datetime.day < birth_datetime.day):
        age -= 1
    
    # Basic education distribution
    education_levels = config['education']['degrees']
    highest_degree = random.choices(
        list(education_levels.keys()),
        weights=list(education_levels.values())
    )[0]
    
    # Build fixed part persona dictionary
    fixed_persona = {
        "basic_info": {
            "name": generate_name(gender),
            "gender": gender,
            "birth_date": birth_date,
            "location": generate_location()
        },
        "age": {
            "current_age": age,
            "latest_date": datetime.now().strftime("%Y-%m-%d")
        },
        "education": {
            "highest_degree": highest_degree,
            "major": "TBC"  # Use original persona as professional background
        },
        "personality": {
            "mbti": random.choice(config['personality']['mbti_types']),
            "tags": []  # Will be generated based on MBTI
        },
        "family_life": generate_family_life(age, birth_date),
        "life_goal": {
            "life_goal_type": random.choice(config['life_goal']['life_goal_types']),
            "statement": "",  # Will be generated based on professional background and personality traits
            "motivation": "Reasons for having such life goals",
            "target_metrics": "",  # Will be generated based on life goals
            # "progress": "Life goals just established"
        }
    }
    
    # Generate skills and interests based on professional background
    if "machine learning" in base_persona.lower() or "ai" in base_persona.lower():
        fixed_persona["education"]["major"] = "Computer Science/Artificial Intelligence"
        fixed_persona["life_goal"]["statement"] = random.choice([
            "Advance AI technology development, achieve more intelligent artificial intelligence systems",
            "Make breakthrough research achievements in machine learning field",
            "Develop revolutionary neural network architectures, promote AGI realization"
        ])
        fixed_persona["life_goal"]["target_metrics"] = random.choice([
            "Publish 10 top conference papers with total impact factor over 100",
            "Develop open-source AI framework used by 1 million developers",
            "Achieve AGI within 20 years, liberate all humanity with AI productivity"
        ])
    
    # Generate personality tags based on MBTI
    fixed_persona["personality"]["tags"] = config['personality']['mbti_tags'][fixed_persona["personality"]["mbti"]]
    
    return fixed_persona


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def validate_and_correct_fixed_persona(base_persona: str, fixed_persona: Dict) -> tuple:
    """Use large language model to validate and correct fixed part persona information"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("[DEBUG] API key not set, returning original fixed_persona")
        return fixed_persona, None, {"token_cost": None}
    
    # Use specialized prompt for step 1
    system_prompt = stage1_1_fixed_prompt
    
    try:
        print("[DEBUG] Sending fixed part correction request to LLM...")

        user_content = f"\nOriginal persona seed: {base_persona}\n\n" + \
                       "Currently generated persona information:\n" + \
                       f"```json\n{json.dumps(fixed_persona, ensure_ascii=False, indent=2)}\n```\n\n" + \
                       "Please analyze and correct the above persona information, " + \
                       "and present the final result only as valid JSON. The JSON must be wrapped " + \
                       "inside a Markdown code block: ```json```."
        
        json_markers = [ 
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_fixed_persona, cost_info = llm_request(
            system_prompt, 
            user_content, 
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(None, cost_info)
        
        print(f"[DEBUG] Successfully processed fixed part with LLM caller")

        if cost_info:
            print(
                f"[DEBUG] Token usage - Input: {cost_info.get('cumulative', {}).get('input_tokens', 'N/A')}, "
                f"Output: {cost_info.get('cumulative', {}).get('output_tokens', 'N/A')}, "
                f"Cost: ${cost_info.get('cumulative', {}).get('total_cost_usd', 'N/A')}"
            )

        return corrected_fixed_persona['Corrected Fixed Part'], cost_info
        
    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        raise  


def generate_uuid_from_seed(persona_seed: str) -> str:
    """Generate deterministic UUID based on persona_seed"""
    # Use SHA256 hash of persona_seed, then convert to UUID
    hash_object = hashlib.sha256(persona_seed.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Use first 16 characters of hash to create UUID
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str


def process_single_persona(persona_seed: str) -> Dict:
    """Process single persona seed, generate fixed part detailed information"""
    print(f"[DEBUG] Processing persona seed: {persona_seed[:50]}...")
    
    # Generate UUID
    persona_uuid = generate_uuid_from_seed(persona_seed)
    print(f"[DEBUG] Generated UUID: {persona_uuid}")
    
    # Step 1: Generate fixed part
    print("[DEBUG] Step 1: Generating fixed part...")
    initial_fixed_persona = generate_fixed_persona(persona_seed)
    corrected_fixed_persona, cost_info = validate_and_correct_fixed_persona(persona_seed, initial_fixed_persona)
    print(f"[DEBUG] Fixed part generation completed")
    
    token_cost = cost_info
    
    # Build result (put fixed part under profile)
    result = {
        "uuid": persona_uuid,
        "profile": {
            "fixed": corrected_fixed_persona
        },
        "metadata": {
            "persona_seed": persona_seed,
            "fixed_generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Add token_cost at the top level, at the end
    if token_cost:
        result["token_cost"] = token_cost
    
    print("[DEBUG] Processing completed")
    return result


def get_persona_seed_at_index(index: int) -> str:
    """Get persona seed at specified index"""
    input_file = os.getenv('STAGE0_FILE_PATH', 'data/stage0_persona.jsonl')
    try:
        with jsonlines.open(input_file) as reader:
            for i, obj in enumerate(reader):
                if i == index:
                    if isinstance(obj, dict) and 'persona' in obj:
                        return obj['persona']
                    break
        return None
    except Exception as e:
        print(f"[DEBUG] Error reading persona seed: {e}:{traceback.format_exc()}")
        return None


def is_seed_exists(output_file: str, persona_seed: str) -> bool:
    """Check if persona seed already exists in output file"""
    if not os.path.exists(output_file):
        return False
        
    try:
        with jsonlines.open(output_file) as reader:
            for item in reader:
                if isinstance(item, dict) and 'metadata' in item and 'persona_seed' in item['metadata']:
                    if item['metadata']['persona_seed'] == persona_seed:
                        return True
    except Exception as e:
        print(f"[DEBUG] Error checking if seed exists: {e}:{traceback.format_exc()}")
    
    return False


def get_random_persona_seed() -> tuple:
    """Randomly get an unprocessed persona seed"""
    input_file = os.getenv('STAGE0_FILE_PATH', 'data/stage0_persona.jsonl')
    output_file = os.getenv('STAGE1_1_FILE_PATH', 'data/stage1_1_fixed.jsonl')
    
    # Get line count of input file
    line_count = 0
    with jsonlines.open(input_file) as reader:
        for _ in reader:
            line_count += 1
    
    if line_count == 0:
        print("[DEBUG] Input file is empty")
        return None, -1
    
    # Load existing seeds
    existing_seeds = set()
    if os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                for item in reader:
                    if isinstance(item, dict) and 'metadata' in item and 'persona_seed' in item['metadata']:
                        existing_seeds.add(item['metadata']['persona_seed'])
        except Exception as e:
            print(f"[DEBUG] Error loading existing seeds: {e}:{traceback.format_exc()}")
    
    # Try random sampling at most 100 times
    max_attempts = 100
    attempts = 0
    
    while attempts < max_attempts:
        random_index = random.randint(0, line_count - 1)
        persona_seed = get_persona_seed_at_index(random_index)
        
        if persona_seed and persona_seed not in existing_seeds:
            print(f"[DEBUG] Successfully randomly sampled unprocessed persona seed, index: {random_index}")
            return persona_seed, random_index
        
        attempts += 1
    
    print(f"[DEBUG] Tried {max_attempts} times, no unprocessed persona seed found")
    return None, -1


def process_single_seed(seed_index: int = 0):
    """Process single persona seed, generate fixed part detailed information and save"""
    input_file = os.getenv('STAGE0_FILE_PATH', 'data/stage0_persona.jsonl')
    output_file = os.getenv('STAGE1_1_FILE_PATH', 'data/stage1_1_fixed.jsonl')
    
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Processing index: {seed_index}")
    
    try:
        # Read persona seed at specified index
        persona_seed = get_persona_seed_at_index(seed_index)
        
        if not persona_seed:
            print(f"[DEBUG] No persona seed found at index {seed_index}")
            return False
            
        print(f"[DEBUG] Processing persona seed: {persona_seed[:100]}...")
        
        # Check if already exists
        if is_seed_exists(output_file, persona_seed):
            print(f"[DEBUG] This persona seed already exists, trying to randomly select another")
            # Randomly select an unprocessed seed
            new_seed, new_index = get_random_persona_seed()
            if new_seed:
                print(f"[DEBUG] Re-selected index: {new_index}")
                return process_single_seed(new_index)
            else:
                print("[DEBUG] No unprocessed persona seed found")
                return False
            
        # Process persona
        detailed_persona = process_single_persona(persona_seed)
        print(f"[DEBUG] Processed persona: {json.dumps(detailed_persona, ensure_ascii=False, indent=2)[:100]}...")
        
        # Save to output file
        with jsonlines.open(output_file, 'a') as writer:
            writer.write(detailed_persona)
            
        print("[DEBUG] Successfully processed and saved persona")
        return True
            
    except Exception as e:
        print(f"Error processing persona: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate fixed part persona information')
    parser.add_argument('--index', type=int, help='Persona index to process (optional)')
    
    args = parser.parse_args()
    
    # If index is specified, use specified index
    if args.index is not None:
        print(f"Using specified index: {args.index}")
        process_single_seed(args.index)
    else:
        # Default to random sampling
        seed, index = get_random_persona_seed()
        if seed:
            print(f"Randomly selected index: {index}")
            process_single_seed(index)
        else:
            print("No available persona seed found")

