import concurrent.futures
import os
from typing import List, Callable, Any

def get_cpu_count() -> int:
    
    try:
        return os.cpu_count() or 4
    except:
        return 4

def parallel_process(items: List[Any], process_func: Callable, stage_name: str = "Unknown") -> List[Any]:
   
    max_workers = get_cpu_count()
    total_count = len(items)
    
    if total_count <= 1:
        
        if total_count == 1:
            print(f"[{stage_name}] Processing 1/1 user")
            return [process_func(items[0])]
        else:
            return []
    
    print(f"[{stage_name}] Start parallel processing of {total_count} users, using {max_workers} threads")
    
    results = [None] * total_count
    completed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        future_to_index = {
            executor.submit(process_func, item): index 
            for index, item in enumerate(items)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                completed_count += 1
                print(f"[{stage_name}] Progress: {completed_count}/{total_count} users have completed")
            except Exception as e:
                print(f"[ERROR] {stage_name} - User {index} processing failed: {e}")
                raise Exception(f"{stage_name}  processing failed, user index: {index}, error: {str(e)}") from e
    
    print(f"[{stage_name}] 所有用户处理完成")
    return results
