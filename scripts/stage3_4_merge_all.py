from datetime import datetime, timedelta
import traceback
import os
import jsonlines
from dotenv import load_dotenv
from typing import Dict, List
import random
import re

from llm_request import calculate_cumulative_cost

load_dotenv()


def _parse_date(date_str: str) -> datetime:
    """Try to parse multiple common date formats; return datetime.max if parsing fails to place at end during sorting."""
    if not date_str:
        return datetime.max
    s = str(date_str).strip()
    if not s:
        return datetime.max

    # Unify separators
    s = s.replace('/', '-').replace('.', '-')

    # Prioritize formats containing time
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    # Handle YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", s):
        try:
            return datetime.strptime(f"{s}-01", '%Y-%m-%d')
        except Exception:
            return datetime.max

    # Handle YYYY
    if re.fullmatch(r"\d{4}", s):
        try:
            return datetime.strptime(f"{s}-01-01", '%Y-%m-%d')
        except Exception:
            return datetime.max

    return datetime.max


def _format_date(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


def _random_date_between(start_dt: datetime, end_dt: datetime) -> str:
    if end_dt < start_dt:
        end_dt = start_dt
    days = (end_dt - start_dt).days
    return _format_date(start_dt + timedelta(days=random.randint(0, max(0, days))))


def read_jsonl(file_path: str) -> List[Dict]:
    arr = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            arr.append(obj)
    return arr


def _reorder_event_fields(event: Dict) -> Dict:
    """Remove unnecessary fields (such as event_id) and reorder fields by event type.
    Rules: Always place event_type, event_name, event_time first; rest by type custom priority, remaining by key name length and dictionary order.
    """
    if not isinstance(event, dict):
        return event

    ev = dict(event)
    # Remove non-universal IDs
    ev.pop('event_id', None)

    event_type = ev.get('event_type', '')

    ordered_keys: List[str] = ['event_index', 'event_type', 'event_name', 'event_time']

    if event_type == 'career_event':
        # Career event stages: short and key information first, long text last
        ordered_keys += [
            'stage_name',
            'main_conflict',
            'stage_result',
            'event_start_time', 'event_end_time', 'user_age',
            'dynamic_updates',
            'stage_description',
            'event_description', 'event_result',
            'related_career_events',
        ]
    elif event_type == 'daily_routine':
        # Preference evolution events
        ordered_keys += [
            'preference_type', 'step',
            'update_direction', 'type_to_update',
            'main_conflict', 'update_reason',
            'before_preference', 'after_preference',
            'causing_event_description',
            'related_daily_routine',
        ]
    elif event_type == 'init_information':
        # Initial information events
        ordered_keys += [
            # If there are dynamic/preference types, try to place them first
            'dynamic_type', 'preference_type',
            'event_description',
            'initial_fixed', 'initial_dynamic', 'initial_preference',
        ]

    result: Dict = {}
    seen = set()
    for k in ordered_keys:
        if k in ev and k not in seen:
            result[k] = ev[k]
            seen.add(k)

    # Append remaining keys: first by length, then by dictionary order
    remaining = [(k, v) for k, v in ev.items() if k not in seen]
    remaining.sort(key=lambda kv: (len(kv[0]), kv[0]))
    for k, v in remaining:
        result[k] = v

    return result


def merge_all(narrative_arc_file: str, preference_file: str, init_events_file: str, output_file: str, regenerate: bool = True):
    print(f"Reading files: {narrative_arc_file}, {preference_file}, {init_events_file}")
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
    
    na = read_jsonl(narrative_arc_file)
    pref = read_jsonl(preference_file)
    initv = read_jsonl(init_events_file)

    by_uuid_na = {x['uuid']: x for x in na if 'uuid' in x}
    by_uuid_pref = {x['uuid']: x for x in pref if 'uuid' in x}
    by_uuid_init = {x['uuid']: x for x in initv if 'uuid' in x}

    common_uuids = set(by_uuid_na) & set(by_uuid_pref) & set(by_uuid_init)
    print(f"[DEBUG] Merging three event streams, UUID intersection: {len(common_uuids)}")

    with jsonlines.open(output_file, 'w') as writer:
        for u in common_uuids:
            na_item = by_uuid_na[u]
            pref_item = by_uuid_pref[u]
            init_item = by_uuid_init[u]

            # 1) core stages
            events: List[Dict] = []
            max_end = datetime.now()
            na_root = na_item.get('narrative_arc', {})
            
            # Add related event indices for career events
            career_event_stages = {}  # Store stage indices for each career event
            
            for ce_idx, ce in enumerate(na_root.get('career_events', [])):
                parent_name = ce.get('event_name')
                if not isinstance(parent_name, str) or not parent_name.strip():
                    raise ValueError(f"Career event missing required 'event_name': {ce}")
                parent_name = parent_name.strip()

                parent_end = ce.get('event_end_time')
                if parent_end:
                    max_end = max(max_end, _parse_date(parent_end))
                
                # Collect all stage indices for current career event
                current_stage_indices = []
                
                # Strictly read stages from current data format; raise if missing/empty
                stages = ce.get('stages')
                if stages is None or not isinstance(stages, list) or len(stages) == 0:
                    raise ValueError(
                        f"Career event '{parent_name}' is missing non-empty 'stages' list in narrative_arc. "
                        f"Got: {type(stages).__name__ if stages is not None else 'None'}"
                    )
                

                # Prepare evenly distributed timeline for career stages using original event times
                raw_start = ce.get('event_start_time')
                raw_end = ce.get('event_end_time')
                if not isinstance(raw_start, str) or not raw_start.strip():
                    raise ValueError(f"Career event '{parent_name}' missing 'event_start_time': {ce}")
                if not isinstance(raw_end, str) or not raw_end.strip():
                    raise ValueError(f"Career event '{parent_name}' missing 'event_end_time': {ce}")

                start_dt = _parse_date(raw_start)
                end_dt = _parse_date(raw_end)
                if start_dt == datetime.max:
                    raise ValueError(f"Career event '{parent_name}' has invalid 'event_start_time': {raw_start}")
                if end_dt == datetime.max:
                    raise ValueError(f"Career event '{parent_name}' has invalid 'event_end_time': {raw_end}")
                if end_dt < start_dt:
                    raise ValueError(f"Career event '{parent_name}' has end time earlier than start time: {raw_start} -> {raw_end}")

                # Ensure minimum span for stages
                min_span_days = max(30, len(stages) * 15)  # At least 15 days per stage
                span_days = max(min_span_days, (end_dt - start_dt).days)
                
                # Adjust end date if needed to ensure minimum span
                if span_days < min_span_days:
                    end_dt = start_dt + timedelta(days=min_span_days)
                
                span_days = max(0, (end_dt - start_dt).days)
                needed_days = max(span_days, len(stages))
                last_offset = -1


                for st_idx, st in enumerate(stages, start=1):
                    # Record current event's actual index in events list
                    current_event_index = len(events)
                    current_stage_indices.append(current_event_index)
                    
                    # Even distribution and strictly increasing within [start_dt, end_dt]
                    # Calculate proper distribution across the actual time span
                    if len(stages) > 1:
                        # Distribute stages evenly across the actual time span
                        progress = (st_idx - 1) / (len(stages) - 1)
                        total_days = (end_dt - start_dt).days
                        day_offset = int(progress * total_days)
                    else:
                        # Single stage, place it in the middle
                        day_offset = (end_dt - start_dt).days // 2
                    
                    day_offset = max(0, min(day_offset, (end_dt - start_dt).days))
                    assigned_dt = start_dt + timedelta(days=day_offset)
                    event_time_val = _format_date(assigned_dt)
                    last_offset = day_offset

                    stage_name = st.get('stage_name')
                    if not isinstance(stage_name, str) or not stage_name.strip():
                        raise ValueError(
                            f"Career event '{parent_name}' stage {st_idx} missing 'stage_name': {st}"
                        )
                    stage_name = stage_name.strip()

                    time_point = st.get('time_point')
                    if not isinstance(time_point, str) or not time_point.strip():
                        raise ValueError(
                            f"Career event '{parent_name}' stage '{stage_name}' missing 'time_point': {st}"
                        )
                    event_time_val = time_point.strip()

                    events.append({
                        # Unified event type
                        'event_type': 'career_event',
                        # Use original parent event name for the first stage, append suffix for later stages
                        'event_name': f"{parent_name} - {stage_name}",
                        # Event occurrence time (evenly sampled like other events)
                        'event_time': event_time_val,
                        # Retain stage information
                        'main_conflict': st.get('main_conflict', ''),
                        'dynamic_updates': st.get('dynamic_updates', []),
                        'stage_description': st.get('stage_description', ''),
                        'stage_result': st.get('stage_result', ''),
                        # New: original event information, avoid stage 4 reverse lookup
                        'event_start_time': ce.get('event_start_time', ''),
                        'event_end_time': ce.get('event_end_time', ''),
                        'user_age': ce.get('user_age', None),
                        'event_description': ce.get('event_description', ''),
                        'event_result': ce.get('event_result', ''),
                        # New: related career event indices
                        'related_career_events': []
                    })
                
                # Store current career event's stage indices
                career_event_stages[ce_idx] = current_stage_indices

            # 2) initial events (start from one month ago, assign different dates to each event)
            init_events = init_item.get('event_list')
            if init_events is None or not isinstance(init_events, list):
                raise ValueError(f"Init events missing valid 'event_list': {init_item}")

            now_dt = datetime.now()
            init_start_dt = now_dt - timedelta(days=30)  # Start from one month ago
            
            for idx, ie in enumerate(init_events):
                ie['event_type'] = 'init_information'
                # Assign different dates to each init event, incrementing from one month ago
                assigned_dt = init_start_dt + timedelta(days=idx)
                ie['event_time'] = _format_date(assigned_dt)
                events.append(ie)

            # 3) preference events (generation time: now~max_end)
            now_dt = datetime.now()
            end_dt = max_end if max_end >= now_dt else now_dt + timedelta(days=365)
            # Preference events by step order of same preference_type, assign strictly increasing dates
            pref_events = pref_item.get('event_list')
            if pref_events is None or not isinstance(pref_events, list):
                raise ValueError(f"Preference events missing valid 'event_list': {pref_item}")
            groups: Dict[str, List[Dict]] = {}
            for pe in pref_events:
                ptype = pe.get('preference_type')
                if not isinstance(ptype, str) or not ptype.strip():
                    raise ValueError(f"Preference event missing 'preference_type': {pe}")
                groups.setdefault(ptype.strip(), []).append(pe)

            # --- FIX START: Interleave events before time allocation to prevent date collision ---
            
            # Step 1: Sort each group internally by step
            for ptype, lst in groups.items():
                try:
                    lst.sort(key=lambda x: x.get('step', 0))
                except Exception:
                    pass
            
            # Step 2: Create a single, interleaved list of all preference events
            all_pref_events: List[Dict] = []
            if groups:
                max_steps = max(len(lst) for lst in groups.values())
                for i in range(max_steps):
                    for ptype, lst in groups.items():
                        if i < len(lst):
                            all_pref_events.append(lst[i])

            # Step 3: Allocate time to the single interleaved list
            if all_pref_events:
                # Define time allocation variables ONCE for all preference events
                base_days = (end_dt - now_dt).days
                # The total number of events is now the length of our interleaved list
                needed_days = max(base_days, len(all_pref_events))
                last_offset = -1 # A single offset tracker for all events

                for idx, pe in enumerate(all_pref_events, start=1):
                    event_name = pe.get('event_name')
                    if not isinstance(event_name, str) or not event_name.strip():
                        raise ValueError(f"Preference event missing 'event_name': {pe}")
                    pe['event_type'] = 'daily_routine'

                    # Even distribution across the entire list, ensuring strict increasing time
                    ideal = int((idx * (needed_days + 1)) / (len(all_pref_events) + 1))
                    day_offset = max(last_offset + 1, ideal)
                    day_offset = min(day_offset, needed_days)
                    assigned_dt = now_dt + timedelta(days=day_offset)

                    pe['event_time'] = _format_date(assigned_dt)
                    last_offset = day_offset
                    
                    # Add related daily_routine events field
                    pe['related_daily_routine'] = []
                    
                    events.append(pe)

            # Sort: parseable dates first, earlier dates first; unparseable placed last
            def _sort_key(ev: Dict) -> tuple:
                dt = _parse_date(ev.get('event_time', ''))
                is_valid = 0 if dt != datetime.max else 1
                return (is_valid, dt)

            events.sort(key=_sort_key)

            # Recalculate related indices after sorting
            # Add related event indices for career events
            # Group by career event base name
            career_groups = {}
            for new_idx, event in enumerate(events):
                if event.get('event_type') == 'career_event':
                    # Extract career event base name (part before -)
                    event_name = event.get('event_name', '')
                    base_name = event_name.split('-')[0] if '-' in event_name else event_name
                    if base_name not in career_groups:
                        career_groups[base_name] = []
                    career_groups[base_name].append(new_idx)
            
            # Add related indices for each career event group
            for base_name, group_indices in career_groups.items():
                for idx in group_indices:
                    if 0 <= idx < len(events):
                        # Add indices of other stages in same group (exclude self)
                        related_indices = [group_idx for group_idx in group_indices if group_idx != idx]
                        events[idx]['related_career_events'] = related_indices

            # Add related event indices for daily_routine events
            # Group by preference_type
            daily_routine_groups = {}
            for new_idx, event in enumerate(events):
                if event.get('event_type') == 'daily_routine':
                    preference_type = event.get('preference_type', '')
                    if preference_type not in daily_routine_groups:
                        daily_routine_groups[preference_type] = []
                    daily_routine_groups[preference_type].append(new_idx)
            
            # Add related indices for each daily_routine group
            for preference_type, group_indices in daily_routine_groups.items():
                for idx in group_indices:
                    if 0 <= idx < len(events):
                        # Add indices of other events in same group (exclude self)
                        related_indices = [group_idx for group_idx in group_indices if group_idx != idx]
                        events[idx]['related_daily_routine'] = related_indices

            # Add event_index to each event before reordering
            for idx, event in enumerate(events):
                event['event_index'] = idx
            
            # Clean and reorder event fields
            ordered_events = [_reorder_event_fields(ev) for ev in events]

            previous_cost = init_item.get('token_cost')
            
            current_stage_cost = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "model": None,
                "pricing_available": False,
                "note": "No LLM calls in this stage - merge operation only"
            }
            
            token_cost = calculate_cumulative_cost(previous_cost, current_stage_cost)
            
            # Clean metadata, only retain necessary fields
            cleaned_metadata = {
                'merge_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_files': {
                    'narrative_arc_file': narrative_arc_file,
                    'preference_file': preference_file,
                    'init_events_file': init_events_file
                },
                'input_data_sources': {
                    'narrative_arc': na_item.get('metadata', {}),
                    'preference_events': pref_item.get('metadata', {}),
                    'init_events': init_item.get('metadata', {})
                },
            }
            
            writer.write({
                'uuid': u,
                'event_list': ordered_events,
                'profile': na_item.get('profile', {}),
                'life_skeleton': na_item.get('life_skeleton', {}),
                'narrative_arc': na_item.get('narrative_arc', {}),
                'metadata': cleaned_metadata,
                'token_cost': token_cost
            })
            print(f"[DEBUG] UUID {u}: merged base event count {len(events)}")


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge all events persona information')
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
    
    narrative_arc_file = os.getenv('STAGE3_1_FILE_PATH', 'data/stage3_1_narrative_arc.jsonl')
    preference_file = os.getenv('STAGE3_2_FILE_PATH', 'data/stage3_2_preference_events.jsonl')
    init_events_file = os.getenv('STAGE3_3_FILE_PATH', 'data/stage3_3_init_events.jsonl')
    output_file = os.getenv('STAGE3_4_FILE_PATH', 'data/stage3_4_merged_events.jsonl')

    for p in [narrative_arc_file, preference_file, init_events_file]:
        if not os.path.exists(p):
            print(f"[ERROR] File does not exist: {p}")
            raise SystemExit(1)

    merge_all(narrative_arc_file, preference_file, init_events_file, output_file, regenerate)




