import os
import copy
import json
import traceback
from typing import Dict, List, Any


with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    config = config['STAGE6']['substage2']


def format_dialogue(dialogue):
    """
    Format dialogue data and convert dialogue rounds into list format
    """
    new_dialogue = []
    
    for i, (_, v) in enumerate(dialogue.items()):
        if not v or len(v) < 3:
            continue
            
        user_talk = copy.deepcopy(v[0])
        bot_talk = copy.deepcopy(v[1])
        timestamp = v[2]["timestamp"]

        user_talk["timestamp"] = timestamp
        bot_talk["timestamp"] = timestamp

        user_talk["dialogue_turn"] = i
        bot_talk["dialogue_turn"] = i

        new_dialogue.append(user_talk)
        new_dialogue.append(bot_talk)

    return new_dialogue


def iter_jsonl(file_path):
    """
    Iteratively read JSONL files
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_single_user(stage5_2_data: dict, stage6_1_data: dict) -> dict:
    """
    Merge data for individual users
    """
    
    assert stage5_2_data["uuid"] == stage6_1_data["uuid"], \
        f"UUID不匹配: {stage5_2_data['uuid']} != {stage6_1_data['uuid']}"
    
    sessions = []
    
    for event in stage5_2_data["event_list"]:
        session = {}
        dialogue_info = event["dialogue_info"]
        
        session["start_time"] = dialogue_info["start_time_point"]
        session["end_time"] = dialogue_info["end_time_point"]
        session["memory_points_count"] = dialogue_info["memory_points_count"]
        session["memory_points"] = dialogue_info["memory_points"]
        session["dialogue_turn_num"] = len(dialogue_info["dialogue"])
        session["dialogue"] = format_dialogue(dialogue_info["dialogue"])
        
        session["dialogue_token_length"] = event.get("dialogue_token_length", 0)
        
        sessions.append(session)
    
    for qa_event in stage6_1_data["event_list"]:
        session_idx = qa_event["event_index"]
        if session_idx < len(sessions):
            sessions[session_idx]["questions"] = qa_event["questions"]
            sessions[session_idx]["question_count"] = qa_event.get("question_count", len(qa_event["questions"]))
    
    total_question_count = stage6_1_data.get("question_count", sum(session.get("question_count", 0) for session in sessions))
    
    new_data = {
        "uuid": stage5_2_data["uuid"],
        "persona_info": stage5_2_data["persona_info"],
        "sessions": sessions,
        "total_dialogue_token_length": stage5_2_data.get("total_dialogue_token_length", 0),
        "total_question_count": total_question_count,
        "token_cost": stage6_1_data.get("token_cost", {})
    }
    
    return new_data


def main(
    stage5_2_output_path: str,
    stage6_1_output_path: str,
    output_file_path: str
):
    """
    Main function: Merge the results of stage5_2 and stage6_1
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Start merging data...")
    print(f"Stage5_2 file: {stage5_2_output_path}")
    print(f"Stage6_1 file: {stage6_1_output_path}")
    print(f"Output file: {output_file_path}")
    
    processed_count = 0
    
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        
        for stage5_2_data, stage6_1_data in zip(
            iter_jsonl(stage5_2_output_path),
            iter_jsonl(stage6_1_output_path)
        ):
            try:
                merged_data = process_single_user(stage5_2_data, stage6_1_data)
                
                f_out.write(json.dumps(merged_data, ensure_ascii=False) + "\n")
                
                processed_count += 1
                print(f"[DEBUG] Processed {processed_count} user data: {merged_data['uuid']}")
                
            except Exception as e:
                print(f"[ERROR] Error processing user data: {e}:{traceback.format_exc()}")
                print(f"[ERROR] Stage5_2 UUID: {stage5_2_data.get('uuid', 'Unknown')}")
                print(f"[ERROR] Stage6_1 UUID: {stage6_1_data.get('uuid', 'Unknown')}")
                continue
    
    print(f"Mission accomplished! Processed {processed_count} user data, the results have been saved to {output_file_path}")


if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Merge the results of stage5_2 and stage6_1')
    parser.add_argument('--stage5-2-file', type=str, 
                       default=config['file_paths']['stage5_2_input_file'],
                       help='Stage5_2 output file path')
    parser.add_argument('--stage6-1-file', type=str,
                       default=config['file_paths']['stage6_1_input_file'], 
                       help='Stage6_1 output file path')
    parser.add_argument('--output-file', type=str, 
                       default=config['file_paths']['output_file'],
                       help='Output file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.stage5_2_file):
        print(f"[ERROR] Stage5_2 file does not exist: {args.stage5_2_file}")
        exit(1)
        
    if not os.path.exists(args.stage6_1_file):
        print(f"[ERROR] Stage6_1 file does not exist: {args.stage6_1_file}")
        exit(1)
    
    main(
        stage5_2_output_path=args.stage5_2_file,
        stage6_1_output_path=args.stage6_1_file,
        output_file_path=args.output_file
    )
