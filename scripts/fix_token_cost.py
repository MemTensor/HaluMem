#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any

def extract_all_token_costs(data: Dict) -> Dict:
    
    token_costs = []
    
    def _extract_recursive(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key == "token_cost" and isinstance(value, dict):
                    token_costs.append({
                        "path": current_path,
                        "data": value
                    })
                else:
                    _extract_recursive(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                _extract_recursive(item, current_path)
    
    _extract_recursive(data)
    return token_costs

def remove_all_token_costs(data: Dict) -> Dict:
    
    def _remove_recursive(obj):
        if isinstance(obj, dict):
            
            new_obj = {}
            for key, value in obj.items():
                if key != "token_cost":
                    new_obj[key] = _remove_recursive(value)
            return new_obj
        elif isinstance(obj, list):
            return [_remove_recursive(item) for item in obj]
        else:
            return obj
    
    return _remove_recursive(data)

def merge_token_costs(token_costs: list) -> Dict:
    
    if not token_costs:
        return None
    
    
    merged_cost = None
    for cost_info in token_costs:
        cost_data = cost_info["data"]
        if "cumulative" in cost_data:
            merged_cost = cost_data
    
    
    if merged_cost is None and token_costs:
        merged_cost = token_costs[-1]["data"]
    
    return merged_cost

def fix_single_file(input_file: str, output_file: str = None) -> bool:
    
    if output_file is None:
        output_file = input_file
    
    print(f"Fix file: {input_file}")
    
    try:
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        token_costs = extract_all_token_costs(data)
        print(f"Found {len(token_costs)} token_cost fields:")
        for cost_info in token_costs:
            print(f"  - {cost_info['path']}")
        
        if len(token_costs) <= 1:
            print("There is only one or no token_cost field in the file, no need to fix it")
            return True
        
        cleaned_data = remove_all_token_costs(data)
        
        merged_cost = merge_token_costs(token_costs)
        
        if merged_cost:
            cleaned_data["token_cost"] = merged_cost
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"Repair completed, the merged token_cost has been added to the top level")
        return True
        
    except Exception as e:
        print(f"Error repairing file {input_file}: {e}")
        return False

def fix_all_stage_files():
    
    stage_files = [
        "data/stage1_1_fixed.jsonl",
        "data/stage1_2_dynamic.jsonl", 
        "data/stage1_3_preferences.jsonl",
        "data/stage2_1_extend_dynamic.jsonl",
        "data/stage2_2_extend_preference.jsonl",
        "data/stage2_3_profile2skeleton.jsonl",
        "data/stage3_1_narrative_arc.jsonl",
        "data/stage3_2_preference_events.jsonl",
        "data/stage3_3_init_events.jsonl",
        "data/stage3_4_merged_events.jsonl",
        "data/stage4_1_events2memories.jsonl",
        "data/stage5_1_dialogue_generation.jsonl",
        "data/stage5_2_token_calculation.jsonl",
        "data/stage6_1_question_generation.jsonl"
    ]
    
    success_count = 0
    total_count = 0
    
    for file_path in stage_files:
        if os.path.exists(file_path):
            total_count += 1
            if fix_single_file(file_path):
                success_count += 1
        else:
            print(f"File not exist: {file_path}")
    
    print(f"\nRepair completed: {success_count}/{total_count} files have been successfully repaired")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix token_cost field issue')
    parser.add_argument('--file', type=str, help='Repair individual files')
    parser.add_argument('--all', action='store_true', help='Fix all stage files')
    
    args = parser.parse_args()
    
    if args.file:
        fix_single_file(args.file)
    elif args.all:
        fix_all_stage_files()
    else:
        print("Please specify the -- file or -- all parameter")
