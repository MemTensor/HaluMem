import json
import os
import copy
import jsonlines
import traceback
from typing import Dict, List
from tqdm import tqdm
import tiktoken

from llm_request import calculate_cumulative_cost


with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE5']['substage2']

# init tokenizer
def load_tokenizer(tokenizer_name: str = "o200k_base"):
    """load tiktoken tokenizer"""
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"[DEBUG] Successfully loaded tiktoken tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        print(f"[ERROR] Failed to load tiktoken tokenizer: {e}:{traceback.format_exc()}")
        raise ValueError(f"Failed to load tiktoken tokenizer {tokenizer_name}: {e}:{traceback.format_exc()}")


def calculate_dialogue_tokens(dialogue_data: dict, tokenizer, event_name: str = "") -> int:
    """
    Calculate the token length of pure text in the conversation
    """
    total_tokens = 0
    dialogue_content = []
    
    for turn_key, turn_data in dialogue_data.items():
        if turn_key.startswith('dialogue_turn_'):
        
            for message in turn_data:
                if isinstance(message, dict) and 'role' in message and 'content' in message:
                    role = message['role']
                    content = message['content']
                    
                    if role in ['user', 'assistant'] and content:
                        tokens = tokenizer.encode(content)
                        token_count = len(tokens)
                        total_tokens += token_count
                        
                        dialogue_content.append(f"[{role}]: {content}")
    
    if event_name:
        print(f"[DEBUG] Event: {event_name}")
        print(f"[DEBUG] Extracted pure dialogue content:")
        for i, content in enumerate(dialogue_content, 1):
            print(f"[DEBUG]   {i}. {content}")
        print(f"[DEBUG] Total number of tokens: {total_tokens}")
        print(f"[DEBUG] " + "="*80)
    
    return total_tokens

def process_single_user(user_data: dict, tokenizer) -> dict:
    """
    Process individual user data and calculate the token length for all conversations
    """
    
    new_data = copy.deepcopy(user_data)
    
    previous_cost = user_data.get('token_cost')
    
    current_stage_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
        "note": "No LLM calls in this stage - token calculation only"
    }
    
    token_cost = calculate_cumulative_cost(previous_cost, current_stage_cost)
    
    total_dialogue_tokens = 0
    
    for event in new_data["event_list"]:

        dialogue_data = event["dialogue_info"]["dialogue"]
        event_tokens = calculate_dialogue_tokens(dialogue_data, tokenizer, event["event_name"])
        
        event["dialogue_token_length"] = event_tokens
        total_dialogue_tokens += event_tokens
    
    new_data["total_dialogue_token_length"] = total_dialogue_tokens
    
    new_data["token_cost"] = token_cost
    
    return new_data

def process_all_users(
    input_file: str,
    output_file: str,
    tokenizer_name: str = "o200k_base",
    regenerate: bool = True
):
    """
    Process all user data and calculate token length
    """
    print(f"Batch processing of files: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Regenerate the pattern: {regenerate}")
    
    tokenizer = load_tokenizer(tokenizer_name)
    
    if regenerate and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[DEBUG] The previous output file has been deleted: {output_file}")
        except Exception as e:
            print(f"[ERROR] Error deleting output file: {e}:{traceback.format_exc()}")
    
    if not regenerate and os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                existing_count = sum(1 for _ in reader)
            if existing_count > 0:
                print(f"[DEBUG] The output file already exists and contains {existing_count} data, skip processing")
                return True
        except Exception as e:
            print(f"[ERROR] Error checking output file: {e}:{traceback.format_exc()}")
    
    try:
        existing_uuids = set()
        if os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                for item in reader:
                    if isinstance(item, dict) and 'uuid' in item:
                        existing_uuids.add(item['uuid'])
        
        processed_count = 0
        skipped_count = 0
        
        with jsonlines.open(input_file) as reader:
            for user_data in tqdm(reader, desc="Processing users"):
                user_uuid = user_data.get('uuid')
                if not user_uuid:
                    print("[DEBUG] Skip user data without UUID")
                    skipped_count += 1
                    continue
                
                if user_uuid in existing_uuids:
                    print(f"[DEBUG] Skip processed users: {user_uuid}")
                    skipped_count += 1
                    continue
                
                result_data = process_single_user(user_data, tokenizer)
                
                with jsonlines.open(output_file, 'a') as writer:
                    writer.write(result_data)
                
                processed_count += 1
                print(f"[DEBUG] Processed {processed_count} user data, total token length: {result_data['total_dialogue_token_length']}")
        
        print(f"[DEBUG] Batch processing completed, processed {processed_count} user data, skipped {skipped_count}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error occurred during batch processing: {e}:{traceback.format_exc()}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate the length of the dialogue token')
    parser.add_argument('--input-file', type=str, default=config['file_paths']['input_file'],
                       help='Input file path')
    parser.add_argument('--output-file', type=str, default=config['file_paths']['output_file'],
                       help='Output file path')
    parser.add_argument('--tokenizer-name', type=str, default="o200k_base",
                       help='tiktoken tokenizer name')
    parser.add_argument('--regenerate', action='store_true', default=True,
                       help='Whether to completely regenerate (default: True)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip existing parts (mutually exclusive with --regenerate)')
    
    args = parser.parse_args()
    
    if args.skip_existing:
        regenerate = False
    else:
        regenerate = args.regenerate
    
    print(f"Generate Mode: {'Regenerate' if regenerate else 'Skip existing ones'}")
    print("Process all user data...")
    process_all_users(args.input_file, args.output_file, args.tokenizer_name, regenerate)