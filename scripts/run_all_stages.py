#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def run_script(python_bin: str, script_path: Path) -> Tuple[bool, float]:
    
    print(f"[RUN] Start: {script_path.name}")
    start_time = time.time()
    
    try:
        subprocess.run([python_bin, str(script_path)], check=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[OK] Finished: {script_path.name} (Time consuming: {execution_time:.2f} s)\n")
        return True, execution_time
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[ERR] Failed: {script_path.name}, returncode={e.returncode} (Time consuming: {execution_time:.2f} s)")
        sys.exit(e.returncode)


def format_time(seconds: float) -> str:
    
    if seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} m {remaining_seconds:.2f} s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} h {minutes} m {remaining_seconds:.2f} s"


def print_time_report(execution_times: List[Tuple[str, float]], total_time: float) -> None:
    print("=" * 80)
    print("📊 Execution Time Statistics Report")
    print("=" * 80)
    
    # 按时间排序（从长到短）
    sorted_times = sorted(execution_times, key=lambda x: x[1], reverse=True)
    
    print(f"Total execution time: {format_time(total_time)}")
    print(f"Number of executed scripts: {len(execution_times)}")
    print()
    
    print("📋 Details of execution time for each script:")
    print("-" * 80)
    print(f"{'Index':<4} {'Script name':<35} {'Execution time':<15} {'Proportion':<10} {'Status'}")
    print("-" * 80)
    
    for idx, (script_name, exec_time) in enumerate(execution_times, 1):
        percentage = (exec_time / total_time * 100) if total_time > 0 else 0
        status = "✅" if exec_time > 0 else "⏭️"
        print(f"{idx:<4} {script_name:<35} {format_time(exec_time):<15} {percentage:>6.1f}% {status}")
    
    print("-" * 80)
    
    if sorted_times:
        print(f"⏱️ The most time-consuming script: {sorted_times[0][0]} ({format_time(sorted_times[0][1])})")
        if len(sorted_times) > 1:
            print(f"⚡ The fastest script: {sorted_times[-1][0]} ({format_time(sorted_times[-1][1])})")
    print()
    print("=" * 80)


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    python_bin = sys.executable or "python3"

    # stage1 -> stage2 -> stage3 -> stage4 -> stage5 -> stage6
    ordered_scripts = [
        "stage1_1_fixed.py",
        "stage1_2_dynamic.py",
        "stage1_3_preferences.py",
        "stage2_1_extend_dynamic.py",
        "stage2_2_extend_preference.py",
        "stage2_3_profile2skeleton.py",
        "stage3_1_skeleton2core.py",
        "stage3_2_skeleton2routine.py",
        "stage3_3_init2events.py",
        "stage3_4_merge_all.py",
        "stage4_1_events2memories.py",
        "stage5_1_dialogue_generation.py",
        "stage5_2_token_calculation.py",
        "stage6_1_question_generation.py",
        "stage6_2_final_result_medium.py",
        "stage6_3_final_result_long.py"
    ]

    print("[INFO] The script will be executed in the following order：")
    for idx, name in enumerate(ordered_scripts, 1):
        print(f"  {idx:02d}. {name}")
    print("")

    execution_times: List[Tuple[str, float]] = []
    total_start_time = time.time()

    for script_name in ordered_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"[WARN] File missing, skip：{script_name}")
            execution_times.append((script_name, 0.0)) 
            continue
        
        success, exec_time = run_script(python_bin, script_path)
        execution_times.append((script_name, exec_time))

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print("[DONE] All script execution has ended.")
    print_time_report(execution_times, total_execution_time)


if __name__ == "__main__":
    main()



