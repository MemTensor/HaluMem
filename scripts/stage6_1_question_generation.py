import json
import random
import os
import copy
import traceback
import jsonlines
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List
from tqdm import tqdm

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
    config = json.load(f)
    config = config['STAGE6']['substage1']

# Load prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")
        raise ValueError(f"Cannot load prompt file {prompt_path}: {e}:{traceback.format_exc()}")

# Load stage6_1 prompt
stage6_1_question_generation_prompt_template = load_prompt(config['prompts']['stage6_1_question_generation'])

# Question category definitions
QUESTION_CATEGORIES = {
    "Basic Fact Recall": {
        "name": "Basic Fact Recall",
        "definition": "Directly ask about single objective facts or user preferences that explicitly appear in the dialogue, without requiring reasoning or information integration.",
        "requirements": [
            "Questions should target information explicitly expressed by the user (such as name, date, location, preferences, etc.)",
            "Answers must be unique and clear, with evidence typically being a single core memory point",
            "Avoid questions requiring reasoning or synthesis of multiple information points",
            "Can ask about: personal information, work information, family members, important dates, clear preferences, etc."
        ],
        "example_directions": [
            "User's basic information (name, age, residence, etc.)",
            "Clearly mentioned preferences (favorite food, color, activities, etc.)",
            "Specific facts (workplace, school name, pet name, etc.)"
        ]
    },
    
    "Multi-hop Inference": {
        "name": "Multi-hop Inference",
        "definition": "Requires synthesizing multiple information fragments from dialogues, and can only derive answers through logical reasoning or temporal reasoning.",
        "requirements": [
            "Must involve at least 2-3 different memory points",
            "Requires time calculation, logical reasoning, or relationship derivation",
            "Answers cannot be directly obtained from a single memory point",
            "Reasoning chain must be clear and reasonable"
        ],
        "example_directions": [
            "Temporal reasoning (calculating specific dates based on multiple time points)",
            "Relationship reasoning (deriving through multiple character relationships)",
            "Conditional combination (comprehensive judgment meeting multiple conditions)",
            "Causal reasoning (deriving results based on multiple events)"
        ]
    },
    
    "Dynamic Update": {
        "name": "Dynamic Update",
        "definition": "Tests the ability to track information changes over time, requiring identification of the latest status or preference changes.",
        "requirements": [
            "Must involve changes in the same information at different time points",
            "Questions should ask about 'current' or 'latest' status",
            "Need to reflect the replacement relationship between old and new information",
            "Can include: location changes, preference updates, status transitions, etc."
        ],
        "example_directions": [
            "Work/residence location changes",
            "Preference evolution (from liking A to liking B)",
            "Health/emotional status changes",
            "Plan or goal adjustments"
        ]
    },
    
    "Memory Boundary": {
        "name": "Memory Boundary",
        "definition": "Tests the system's ability to identify unknown information by asking about details not mentioned in the input information to examine whether the system will fabricate answers.",
        "requirements": [
            "Must ask about specific information that cannot be found in any memory points",
            "Questions should appear reasonable and natural, avoiding overly obvious traps",
            "Answers must clearly acknowledge that information is unknown or not provided by the user",
            "Evidence list must be an empty list []"
        ],
        "example_directions": [
            "Ask about personal details never mentioned (such as middle name, blood type, etc.)",
            "Ask about family member information never discussed",
            "Ask about specific but unprovided numerical information (such as specific salary, weight, etc.)",
            "Ask about historical events or experiences not covered"
        ]
    },
    
    "Generalization & Application": {
        "name": "Generalization & Application",
        "definition": "Based on known user preferences or characteristics, infer reasonable suggestions or judgments in new scenarios.",
        "requirements": [
            "Not directly asking about known information, but applying to new situations",
            "Need to extract general principles from specific preferences",
            "Evidence is user attribute memory points that support this generalization logic",
            "Answers are generalized applications based on evidence, reflecting personalization and reasonable inference",
            "Avoid over-generalization or unreasonable inference"
        ],
        "example_directions": [
            "Recommend new restaurants based on dietary preferences",
            "Suggest office environment based on work habits",
            "Infer possible choices from past experiences",
            "Predict behavioral tendencies based on personality traits"
        ]
    },
    
    "Memory Conflict": {
        "name": "Memory Conflict",
        "definition": "Tests the system's ability to identify and correct erroneous premises. Questions deliberately contain incorrect information that directly contradicts known memory points, requiring the system to identify contradictions, correct errors, and answer based on correct information.",
        "requirements": [
            "Questions must embed an erroneous premise that contradicts core memory points",
            "Answers should first correct the erroneous premise, then supplement correct information",
            "Must not ignore or default to erroneous premises, must explicitly point out conflicts",
            "Evidence list must only contain core memory point original text that can directly refute the erroneous premise",
            "Answers must avoid fabricating or guessing unknown information"
        ],
        "example_directions": [
            "Incorrectly cite user preferences or limitations",
            "Confuse timeline or event sequence",
            "Wrong character relationship assumptions",
            "Premises contrary to what the user explicitly expressed"
        ]
    }
}

# Configuration constants
SEED = config.get('seed', 2025)
MIN_QUESTION_NUM = config.get('min_question_num', 3)
MIN_QUESTION_NUM_PER_STAGE = config.get('min_question_num_per_stage', 1)
MAX_QUESTION_NUM_PER_STAGE = config.get('max_question_num_per_stage', 3)
MIN_QUESTION_TYPE_NUM = config.get('min_question_type_num', 2)
MAX_QUESTION_TYPE_NUM = config.get('max_question_type_num', 4)

QUESTION_TYPE_FOR_INIT_INFORMATION = config.get('question_type_for_init_information', 
    ["Basic Fact Recall", "Memory Boundary", "Memory Conflict"])
QUESTION_TYPE_FOR_DAILY_ROUTINE = config.get('question_type_for_daily_routine', 
    ["Basic Fact Recall", "Memory Boundary", "Generalization & Application", "Memory Conflict"])
QUESTION_TYPE_FOR_CAREER_EVENT = config.get('question_type_for_career_event', 
    ['Basic Fact Recall', 'Multi-hop Inference', 'Dynamic Update', 'Memory Boundary', 'Generalization & Application', 'Memory Conflict'])

random.seed(SEED)


def allocate_questions(stage_count, question_types):
    """
    Allocate question numbers and types for events
    
    Args:
        stage_count: Number of event stages
        question_types: List of available question types
        
    Returns:
        list: List of [(question_type, question_count), ...]
    """
    # Step 1: Calculate total number of questions
    alpha = random.random() * (MAX_QUESTION_NUM_PER_STAGE - MIN_QUESTION_NUM_PER_STAGE) + MIN_QUESTION_NUM_PER_STAGE  # α ∈ [MIN_QUESTION_NUM_PER_STAGE, MAX_QUESTION_NUM_PER_STAGE]
    Q = MIN_QUESTION_NUM + alpha * (stage_count - 1)
    Q = round(Q)  # Ensure integer

    # Step 2: Randomly select question types
    k = min(random.randint(MIN_QUESTION_TYPE_NUM, MAX_QUESTION_TYPE_NUM), Q)
    chosen_types = random.sample(question_types, k)

    # Step 3: Initialize at least 1 question per type
    allocation = {t: 1 for t in chosen_types}
    remaining = Q - k

    # Step 4: Allocate remaining questions by random weights
    weights = {t: random.uniform(1, 3) for t in chosen_types}
    total_w = sum(weights.values())

    for t in chosen_types:
        extra = round(remaining * (weights[t] / total_w))
        allocation[t] += extra

    # Step 5: Correct differences caused by rounding
    while sum(allocation.values()) < Q:
        allocation[random.choice(chosen_types)] += 1
    while sum(allocation.values()) > Q:
        # Ensure quantity doesn't go to 0
        t = random.choice(chosen_types)
        if allocation[t] > 1:
            allocation[t] -= 1

    # Step 6: Convert to specified format output
    result = [(t, allocation[t]) for t in chosen_types]
    return result


def generate_question_type_description(category_key):
    """
    Generate complete category description text based on category key
    
    Args:
        category_key: Key value of question category
        
    Returns:
        str: Formatted category description text
    """
    if category_key not in QUESTION_CATEGORIES:
        raise ValueError(f"Unknown question category: {category_key}")
    
    category = QUESTION_CATEGORIES[category_key]
    
    description = f"""Question Category: {category['name']}

Definition: {category['definition']}

Specific Requirements:
{chr(10).join(f"- {req}" for req in category['requirements'])}

Example Directions:
{chr(10).join(f"- {example}" for example in category['example_directions'])}"""
    
    return description


def generate_dialogue_format_info(dialogue_info: dict, dislogue_index: int = 1):
    """
    Generate dialogue information
    """
    start_time = dialogue_info['start_time_point']
    end_time = dialogue_info['end_time_point']
    dialogue_goal = dialogue_info['dialogue_goal']
    dialogue_summary = dialogue_info['dialogue_summary']
    memory_points = '\n' + '\n'.join(
        [
            f"  - [{i['timestamp']}][{i['memory_type']}]{i['memory_content']}" for i in dialogue_info['memory_points'] if i["memory_source"] != "interference"
        ]
    )
    dialogue_format_info = f"## Dialogue {dislogue_index}\n\n" + \
                           f"* `Start Time`: {start_time}\n" + \
                           f"* `End Time`: {end_time}\n" + \
                           f"* `Dialogue Goal`: {dialogue_goal}\n" + \
                           f"* `Dialogue Summary`: {dialogue_summary}\n" + \
                           f"* `Memory Points`: {memory_points}" + "\n\n"
    
    return dialogue_format_info


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

def generate_questions_for_event(prompt: str, previous_cost: Dict = None) -> Dict:
    """Generate questions for a single event"""
    if not os.getenv('OPENAI_API_KEY'):
        print("[DEBUG] API key not set, returning default questions")
        return {
            "questions": [
                {
                    "question": "Default question",
                    "answer": "Default answer",
                    "evidence": [],
                    "difficulty": "easy"
                }
            ],
            "token_cost": calculate_cumulative_cost(previous_cost, {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "total_cost_usd": 0.0, "pricing_available": False
            })
        }
    
    try:
        print("[DEBUG] Sending question generation request to LLM...")

        # Find JSON part in chain-of-thought output
        markers = ["Question generation result", "Generated questions", "JSON result", "Final JSON", "Complete JSON", "Generation result"]

        raw_response, cost_info = llm_request("", prompt, json_markers=markers)
        
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
            questions_result = json.loads(json_content)
            print(f"[DEBUG] Successfully parsed question generation JSON")
            questions_result["token_cost"] = token_cost
            return questions_result
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}:{traceback.format_exc()}, trying to extract using regex")
            # Try to extract JSON using stricter method
            import re
            json_pattern = r'({[\s\S]*})'
            matches = re.findall(json_pattern, json_content)
            for match in matches:
                try:
                    questions_result = json.loads(match)
                    print(f"[DEBUG] Successfully extracted question generation JSON through regex")
                    questions_result["token_cost"] = token_cost
                    return questions_result
                except json.JSONDecodeError:
                    print(f"[DEBUG] Content extracted by regex is not valid JSON")
            
            raise
        
    except Exception as e:
        print(f"[ERROR] Question generation failed: {e}:{traceback.format_exc()}")
        raise


def process_single_user(user_data: dict):
    """
    Stage 6.1: Generate questions for each event
    
    Args:
        user_data: User data containing event list and token length information
        
    Returns:
        dict: User data containing generated questions
    """
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
    
    new_data = copy.deepcopy(user_data)
    new_data["event_list"] = []
    # Step 1: Merge events
    merged_events = []
    processed_event_idxs = set()

    for idx in range(len(events) - 1, -1, -1):
        if idx in processed_event_idxs:
            continue
        
        event = events[idx]
        new_event = copy.deepcopy(event)
        new_event["event_index"] = idx

        dialogue_format_info_list = []
        if event["event_type"] == "career_event":
            new_event["event_name"] = event["event_name"].split("-")[0]
            new_event["event_start_time"] = event["event_start_time"]
            new_event["event_end_time"] = event["event_end_time"]
            stage_idxs = event.get("related_career_events", [])
            stage_idxs.append(idx)

            for ii, sidx in enumerate(stage_idxs, 1):
                if 0 <= sidx < len(events):
                    stage = events[sidx]
                    dialogue_format_info = generate_dialogue_format_info(stage["dialogue_info"], ii)
                    dialogue_format_info_list.append(dialogue_format_info)

            processed_event_idxs.update(stage_idxs)
            new_event["event_stage_count"] = len(stage_idxs)
            new_event["included_events_idxs"] = stage_idxs
        else:
            new_event["event_time"] = event["event_time"]
            new_event["event_stage_count"] = 1
            new_event["included_events_idxs"] = [idx]
            dialogue_format_info = generate_dialogue_format_info(event["dialogue_info"])
            dialogue_format_info_list.append(dialogue_format_info)
            processed_event_idxs.add(idx)

        new_event["dialogue_format_info_list"] = dialogue_format_info_list
        merged_events.append(new_event)

    merged_events = merged_events[::-1]
    
    # Step 2: Generate questions
    for event in tqdm(merged_events, desc="Generating Questions"):
        event["questions"] = []

        event_type = event["event_type"]
        if event_type == "career_event":
            question_type_list = QUESTION_TYPE_FOR_CAREER_EVENT
        elif event_type == "daily_routine":
            question_type_list = QUESTION_TYPE_FOR_DAILY_ROUTINE
        elif event_type == "init_information":
            question_type_list = QUESTION_TYPE_FOR_INIT_INFORMATION
        else:
            raise ValueError(f"Unknown event type: {event_type}")
        
        event_stage_count = event["event_stage_count"]
        question_type_num = allocate_questions(event_stage_count, question_type_list)
        dialogue_info = "".join(event["dialogue_format_info_list"])

        for question_type, question_num in question_type_num:

            question_type_description = generate_question_type_description(question_type)
            prompt = stage6_1_question_generation_prompt_template.format(
                question_num=question_num,
                question_type=question_type,
                dialogue_info=dialogue_info,
                question_type_description=question_type_description
            )

            questions = generate_questions_for_event(prompt, previous_cost)
            
            if "token_cost" in questions and questions["token_cost"]:
                current_cost = questions["token_cost"].get("current_stage", {})
                current_stage_total_cost["input_tokens"] += current_cost.get("input_tokens", 0)
                current_stage_total_cost["output_tokens"] += current_cost.get("output_tokens", 0)
                current_stage_total_cost["total_tokens"] += current_cost.get("total_tokens", 0)
                current_stage_total_cost["total_cost_usd"] += current_cost.get("total_cost_usd", 0.0)
                if current_stage_total_cost["model"] is None:
                    current_stage_total_cost["model"] = current_cost.get("model")
                current_stage_total_cost["pricing_available"] = current_cost.get("pricing_available", False)
            
            for q in questions["questions"]:
                q["question_type"] = question_type
                event["questions"].append(q)

        event["question_count"] = len(event["questions"])
        new_data["event_list"].append(event)
    
    final_token_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
    
    total_question_count = sum(event["question_count"] for event in new_data["event_list"])
    
    new_data["token_cost"] = final_token_cost
    new_data["question_count"] = total_question_count
    
    return new_data


def process_all_users(
    input_file: str,
    output_file: str,
    regenerate: bool = True
):
    """
    Process all user data and generate questions
    
    Args:
        input_file: Input file path
        output_file: Output file path
        regenerate: Whether to regenerate
    """
    print(f"Batch processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Regeneration mode: {regenerate}")
    
    # If in regeneration mode, delete previous output file
    if regenerate and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[DEBUG] Deleted previous output file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Error deleting output file: {e}:{traceback.format_exc()}")
    
    # If not in regeneration mode, check if output file exists and contains data
    if not regenerate and os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                existing_count = sum(1 for _ in reader)
            if existing_count > 0:
                print(f"[DEBUG] Output file already exists and contains {existing_count} records, skipping processing")
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
                    print("[DEBUG] Skipping user data missing UUID")
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
                result_data = process_single_user(user_data)
                print(f"[DEBUG] Successfully processed user {user_data.get('uuid')}")
                return result_data
            except Exception as e:
                print(f"[ERROR] Error processing user {user_data.get('uuid')}: {e}")
                return None
        
        processed_results = parallel_process(all_user_data, process_single_user_data, "Stage6.1-QuestionGeneration")
        
        valid_results = [result for result in processed_results if result is not None]
        with jsonlines.open(output_file, 'w') as writer:
            for result in valid_results:
                writer.write(result)
        
        print(f"[DEBUG] Batch processing completed, processed {len(valid_results)} user data, skipped {skipped_count}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during batch processing: {e}:{traceback.format_exc()}")
        return False


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate question content')
    parser.add_argument('--input-file', type=str, default=config['file_paths']['input_file'],
                       help='Input file path')
    parser.add_argument('--output-file', type=str, default=config['file_paths']['output_file'],
                       help='Output file path')
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
    print("Processing all user data...")
    process_all_users(args.input_file, args.output_file, regenerate)