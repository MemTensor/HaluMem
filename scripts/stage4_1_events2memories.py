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

# Import functions and classes from utils
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ConversationConverter

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


class ProfileUpdater:
    """Profile dynamic updater, used to apply dynamic updates from events to profile"""
    
    @staticmethod
    def apply_dynamic_updates(profile: Dict, dynamic_updates: List[Dict]) -> Dict:
        """
        Apply dynamic updates to profile
        
        Args:
            profile: Current profile data
            dynamic_updates: Dynamic update list
            
        Returns:
            Updated profile data
        """
        if not dynamic_updates:
            return profile
        
        # Deep copy profile to avoid modifying original data
        updated_profile = json.loads(json.dumps(profile))
        
        for update in dynamic_updates:
            type_to_update = update.get("type_to_update", "")
            after_dynamic = update.get("after_dynamic", {})
            
            if not type_to_update or not after_dynamic:
                continue
            
            # Ensure dynamic field exists
            if "dynamic" not in updated_profile:
                updated_profile["dynamic"] = {}
            
            # Apply updates based on update type
            if type_to_update == "career_status":
                # Directly replace career_status, because after_dynamic contains complete status
                updated_profile["dynamic"]["career_status"] = after_dynamic
            elif type_to_update == "health_status":
                # Directly replace health_status
                updated_profile["dynamic"]["health_status"] = after_dynamic
            elif type_to_update == "social_relationships":
                # Directly replace social_relationships
                updated_profile["dynamic"]["social_relationships"] = after_dynamic
            elif type_to_update == "work_experience":
                if "work_experience" not in updated_profile["dynamic"]:
                    updated_profile["dynamic"]["work_experience"] = {}
                updated_profile["dynamic"]["work_experience"].update(after_dynamic)
            elif type_to_update == "skills":
                if "skills" not in updated_profile["dynamic"]:
                    updated_profile["dynamic"]["skills"] = {}
                updated_profile["dynamic"]["skills"].update(after_dynamic)
            elif type_to_update == "achievements":
                if "achievements" not in updated_profile["dynamic"]:
                    updated_profile["dynamic"]["achievements"] = {}
                updated_profile["dynamic"]["achievements"].update(after_dynamic)
        
        print(f"[DEBUG] Applied dynamic updates completed, updated {len(dynamic_updates)} fields")
        return updated_profile
    
    @staticmethod
    def apply_preference_updates(profile: Dict, event: Dict) -> Dict:
        """
        Apply preference updates to profile
        
        Args:
            profile: Current profile data
            event: Event containing preference updates
            
        Returns:
            Updated profile data
        """
        if event.get("event_type") != "daily_routine":
            return profile
        
        # Deep copy profile to avoid modifying original data
        updated_profile = json.loads(json.dumps(profile))
        
        # Ensure preferences field exists
        if "preferences" not in updated_profile:
            updated_profile["preferences"] = {}
        
        preference_type = event.get("preference_type", "")
        after_preference = event.get("after_preference", {})
        
        if preference_type and after_preference:
            # Directly use after_preference to replace corresponding preference type
            updated_profile["preferences"][preference_type] = after_preference
        
        return updated_profile

# Memory point template mapping table - based on template structure in utils.py
MEMORY_TEMPLATES = {
    # career_status related templates
    "career_status.employment_status": "{persona_name}'s employment status is {value}",
    "career_status.industry": "{persona_name} works in the {value} industry",
    "career_status.company_name": "{persona_name} works at {value}",
    "career_status.job_title": "{persona_name}'s job title is {value}",
    "career_status.monthly_income": "{persona_name}'s monthly income is {value} yuan",
    "career_status.savings_amount": "{persona_name}'s savings amount is {value} yuan",
    
    # health_status related templates
    "health_status.physical_health": "{persona_name}'s physical health status: {value}",
    "health_status.mental_health": "{persona_name}'s mental health status: {value}",
    "health_status.physical_chronic_conditions": "{persona_name}'s chronic disease situation: {value}",
    "health_status.mental_chronic_conditions": "{persona_name}'s mental illness situation: {value}",
    
    # social_relationships related templates
    "social_relationships": "{persona_name}'s {relationship_type} {name}, {description}",
    
    # work_experience related templates
    "work_experience": "{persona_name}'s work experience: {description}",
    
    # skills related templates
    "skills": "{persona_name}'s skills: {description}",
    
    # achievements related templates
    "achievements": "{persona_name}'s achievements: {description}",
}

# Field name mapping table - used to generate more natural memory point content
FIELD_NAME_MAPPINGS = {
    "company_name": "work company",
    "job_title": "job title",
    "monthly_income": "monthly income",
    "savings_amount": "savings amount",
    "employment_status": "employment status",
    "industry": "industry",
    "physical_health": "physical health status",
    "mental_health": "mental health status",
    "physical_chronic_conditions": "chronic disease situation",
    "mental_chronic_conditions": "mental illness situation",
}

# Status mapping table
STATUS_MAPPINGS = {
    "employment_status": {
        'employed': 'employed',
        'unemployed': 'unemployed',
        'student': 'student',
        'retired': 'retired'
    }
}


class UpdateTracker:
    """System memory point update tracker"""
    
    @staticmethod
    def generate_original_memory(field_path: str, before_value: any, persona_name: str, type_to_update: str = None) -> str:
        """
        Generate original memory point content based on field path and value before update
        
        Args:
            field_path: Field path, such as "career_status.company_name"
            before_value: Value before update
            persona_name: Character name
            type_to_update: Update type, for special processing
            
        Returns:
            Original memory point content
        """
        if not before_value or before_value == "":
            return ""
        
        # Extract field name
        field_name = field_path.split('.')[-1] if '.' in field_path else field_path
        
        # Handle special field status mapping
        if field_name in STATUS_MAPPINGS:
            before_value = STATUS_MAPPINGS[field_name].get(before_value, before_value)
        
        # Find corresponding template
        template_key = field_path
        if template_key in MEMORY_TEMPLATES:
            template = MEMORY_TEMPLATES[template_key]
            return template.format(persona_name=persona_name, value=before_value)
        
        # Use field name mapping to generate more natural memory points
        if field_name in FIELD_NAME_MAPPINGS:
            field_display_name = FIELD_NAME_MAPPINGS[field_name]
            return f"{persona_name}'s {field_display_name} is {before_value}"
        
        # If no mapping found, use generic format
        return f"{persona_name}'s {field_name} is {before_value}"
    
    @staticmethod
    def track_system_memory_update(system_memory: Dict, before_dynamic: Dict, 
                                  changed_keys: List[str], persona_name: str, 
                                  type_to_update: str) -> Dict:
        """
        Track system memory point updates, fill original_memories
        
        Args:
            system_memory: System memory point
            before_dynamic: Dynamic data before update
            changed_keys: List of changed fields
            persona_name: Character name
            type_to_update: Update type
            
        Returns:
            Updated system memory point
        """
        original_memories = []
        
        # Parse updated fields from memory_content
        content = system_memory.get("memory_content", "")
        
        # Generate original memory points based on changed_keys
        for key in changed_keys:
            if key in before_dynamic:
                before_value = before_dynamic[key]
                
                # Build field path
                field_path = f"{type_to_update}.{key}"
                
                # Generate original memory point
                original_memory = UpdateTracker.generate_original_memory(
                    field_path, before_value, persona_name, type_to_update
                )
                
                if original_memory:
                    original_memories.append(original_memory)
        
        # Update system memory point
        system_memory["original_memories"] = original_memories
        
        # Post-processing mechanism: if is_update is "True" but original_memories is empty, set is_update to "False"
        if system_memory.get("is_update") == "True" and not original_memories:
            system_memory["is_update"] = "False"
            print(f"[DEBUG] Post-processing: changed is_update from 'True' to 'False' because original_memories is empty")
        
        return system_memory


# Load configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    full_config = json.load(f)
    config = full_config['STAGE4']['substage1']

# Read prompt from file
def load_prompt(prompt_path: str) -> str:
    """Load prompt file from specified path"""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to load prompt file {prompt_path}: {e}:{traceback.format_exc()}")
        raise ValueError(f"Cannot load prompt file {prompt_path}: {e}:{traceback.format_exc()}")

# Load stage4_1 prompt
stage4_1_events2memories_prompt = load_prompt(config['prompts']['stage4_1_events2memories'])


def extract_profile_info(data: dict) -> str:
    """Extract and format character information"""
    if not data:
        raise ValueError("Input data is empty, cannot extract character information")
    
    # Check if data is already in profile structure or contains profile structure
    if "profile" in data:
        # data contains profile field, such as user_data structure
        profile = data.get("profile", {})
        if not profile:
            raise ValueError("profile field is empty, cannot extract character information")
    else:
        # data itself is the profile structure, such as persona_data structure
        profile = data
    
    fixed = profile.get("fixed", {})
    if not fixed:
        raise ValueError("fixed field is empty, cannot extract character information")
    
    parts = []

    # Record time
    recorded_time = fixed.get("age", {}).get('latest_date')
    if recorded_time:
        try:
            recorded_time = datetime.strptime(recorded_time, "%Y-%m-%d").strftime("%b %d, %Y")
            parts.append(f"[Recorded on {recorded_time}]")
        except Exception as e:
            print(f"[ERROR] Time formatting failed: {e}:{traceback.format_exc()}")
    
    # Basic information
    basic_info = fixed.get("basic_info", {})
    if not basic_info:
        raise ValueError("basic_info field is empty, cannot extract character information")
    
    if basic_info.get('name'):
        parts.append(f"Name: {basic_info.get('name')};")
    else:
        raise ValueError("Character name missing, cannot extract character information")
    
    if basic_info.get('gender'):
        parts.append(f"Gender: {basic_info.get('gender')};")
    if basic_info.get('birth_date'):
        parts.append(f"Birth Date: {basic_info.get('birth_date')};")
    if basic_info.get('location'):
        parts.append(f"Location: {basic_info.get('location')};")
    
    # Age
    age = fixed.get("age", {})
    if age.get('current_age'):
        parts.append(f"Current Age: {age.get('current_age')};")
    
    # Education
    education = fixed.get("education", {})
    if education.get('highest_degree'):
        parts.append(f"Highest Degree: {education.get('highest_degree')};")
    if education.get('major'):
        parts.append(f"Major: {education.get('major')};")
    
    # Personality
    personality = fixed.get("personality", {})
    if personality.get('mbti'):
        parts.append(f"MBTI: {personality.get('mbti')};")
    tags = personality.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)};")
    
    # Life goal
    life_goal = fixed.get("life_goal", {})
    if life_goal.get('life_goal_type'):
        parts.append(f"Life Goal Type: {life_goal.get('life_goal_type')};")
    if life_goal.get('statement'):
        parts.append(f"Statement: {life_goal.get('statement')};")
    if life_goal.get('motivation'):
        parts.append(f"Motivation: {life_goal.get('motivation')};")
    if life_goal.get('target_metrics'):
        parts.append(f"Target Metrics: {life_goal.get('target_metrics')}.")
    
    result = " ".join(parts)
    if not result.strip():
        raise ValueError("Extracted character information is empty")
    
    return result


def extract_current_profile_info(data: dict) -> str:
    """Extract and format current profile information, including dynamic updates"""
    if not data:
        raise ValueError("Input data is empty, cannot extract character information")
    
    # Check if data is already in profile structure or contains profile structure
    if "profile" in data:
        # data contains profile field, such as user_data structure
        profile = data.get("profile", {})
        if not profile:
            raise ValueError("profile field is empty, cannot extract character information")
    else:
        # data itself is the profile structure, such as persona_data structure
        profile = data
    
    # Get basic information
    basic_info = extract_profile_info(data)
    
    # Add dynamic information
    dynamic_parts = []
    dynamic = profile.get("dynamic", {})
    
    # Career status
    career_status = dynamic.get("career_status", {})
    if career_status:
        # Check if it's the updated structure (directly contains fields) or the original structure (contains init)
        if isinstance(career_status, dict):
            if "init" in career_status:
                # Original structure, use init
                current_career = career_status.get("init", {})
            else:
                # Updated structure, use directly
                current_career = career_status
        else:
            current_career = career_status
        
        if current_career.get("employment_status"):
            dynamic_parts.append(f"Employment: {current_career.get('employment_status')};")
        if current_career.get("industry"):
            dynamic_parts.append(f"Industry: {current_career.get('industry')};")
        if current_career.get("company_name"):
            dynamic_parts.append(f"Company: {current_career.get('company_name')};")
        if current_career.get("job_title"):
            dynamic_parts.append(f"Job Title: {current_career.get('job_title')};")
        if current_career.get("monthly_income"):
            dynamic_parts.append(f"Monthly Income: {current_career.get('monthly_income')} yuan;")
    
    # Health status
    health_status = dynamic.get("health_status", {})
    if health_status:
        if isinstance(health_status, dict):
            if "init" in health_status:
                current_health = health_status.get("init", {})
            else:
                current_health = health_status
        else:
            current_health = health_status
        
        if current_health.get("physical_health"):
            dynamic_parts.append(f"Physical Health: {current_health.get('physical_health')};")
        if current_health.get("mental_health"):
            dynamic_parts.append(f"Mental Health: {current_health.get('mental_health')};")
    
    # Social relationships
    social_relationships = dynamic.get("social_relationships", {})
    if social_relationships:
        if isinstance(social_relationships, dict):
            if "init" in social_relationships:
                current_relationships = social_relationships.get("init", {})
            else:
                current_relationships = social_relationships
        else:
            current_relationships = social_relationships
        
        if current_relationships:
            relationship_names = list(current_relationships.keys())
            if relationship_names:
                dynamic_parts.append(f"Key Relationships: {', '.join(relationship_names)};")
    
    # Preference information
    preferences = profile.get("preferences", {})
    if preferences:
        preference_types = list(preferences.keys())
        if preference_types:
            dynamic_parts.append(f"Preferences: {', '.join(preference_types)};")
    
    # Combine all information
    if dynamic_parts:
        return f"{basic_info} Current Status: {' '.join(dynamic_parts)}"
    else:
        return basic_info


def generate_utils_results(persona_data: Dict, user_uuid: str):
    """
    Generate results from utils for conversations and memory points (internal use only)
    
    Args:
        persona_data: Character data
        user_uuid: User UUID
        
    Returns:
        Generated utils results dictionary
    """
    try:
        print(f"[DEBUG] Generating utils results...")
        
        # Validate persona_data structure
        if not isinstance(persona_data, dict):
            print(f"[ERROR] persona_data is not a dictionary type: {type(persona_data)}")
            return None
            
        # Ensure preferences field exists and is in correct format
        if 'preferences' not in persona_data:
            print(f"[ERROR] persona_data missing preferences field, adding empty preferences")
            persona_data['preferences'] = {}
        elif not isinstance(persona_data['preferences'], dict):
            print(f"[ERROR] preferences field is not a dictionary type, resetting to empty dictionary")
            persona_data['preferences'] = {}
        
        # Create ConversationConverter instance
        converter = ConversationConverter()
        
        # Generate conversations and memory points using new method
        result = converter.convert_single_persona_and_generate_memories(persona_data, user_uuid)
        
        print(f"[DEBUG] utils results generated successfully")
        return result
        
    except Exception as e:
        print(f"[ERROR] Error generating utils results: {e}:{traceback.format_exc()}")
        return None


def map_range(x, in_min, in_max, out_min, out_max):
    """Map a value from one range to another"""
    return round(out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min))


def add_seconds(date_str: str, seconds: int) -> str:
    """Add seconds to date string"""
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y, %H:%M:%S")
        new_dt = dt + timedelta(seconds=seconds)
        return new_dt.strftime("%b %d, %Y, %H:%M:%S")
    except Exception as e:
        print(f"[ERROR] Time calculation failed: {e}:{traceback.format_exc()}")
        return date_str


def require_end_time_point(end_time_point: str) -> str:
    """Require end_time_point to be provided, raise error if None"""
    if not end_time_point:
        raise ValueError("end_time_point is required")
    return end_time_point

def convert_event_time_to_dialogue_time(event_time: str, memory_count: int) -> tuple:
    """
    Convert event_time (YYYY-MM-DD) to dialogue start and end time based on memory count.

    Args:
        event_time (str): Event time in format "YYYY-MM-DD"
        memory_count (int): Number of memory points

    Returns:
        tuple: (start_time, end_time) in format "MMM DD, YYYY, HH:MM:SS"
    """
    try:
        event_dt = datetime.strptime(event_time, "%Y-%m-%d")
        cfg = config["dialogue_time_settings"]
        mem_range = config["memory_count_range"]

        start_hour = random.randint(cfg["min_start_hour"], cfg["max_start_hour"] - 1)
        start_minute = random.randint(0, 59)
        start_second = random.randint(0, 59)
        start_time = event_dt.replace(hour=start_hour, minute=start_minute, second=start_second)

        if memory_count <= mem_range[0]:
            base_duration_min = cfg["min_duration_min"]
        elif memory_count >= mem_range[1]:
            base_duration_min = cfg["max_duration_min"]
        else:
            ratio = (memory_count - mem_range[0]) / (mem_range[1] - mem_range[0])
            base_duration_min = cfg["min_duration_min"] + ratio * (
                cfg["max_duration_min"] - cfg["min_duration_min"]
            )

        extra_minutes = random.randint(cfg["random_extra_min_low"], cfg["random_extra_min_high"])
        total_duration_min = min(base_duration_min + extra_minutes, cfg["max_duration_min"])

        end_time = start_time + timedelta(minutes=total_duration_min)

        start_time_str = start_time.strftime("%b %d, %Y, %H:%M:%S")
        end_time_str = end_time.strftime("%b %d, %Y, %H:%M:%S")

        return start_time_str, end_time_str

    except Exception as e:
        print(f"[ERROR] Event time conversion failed: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Event time conversion failed: {e}")


def generate_system_memory_points(event: Dict, current_profile: Dict, end_time_point: str = None) -> List[Dict]:
    """Generate system memory points, directly convert update items from events"""
    print(f"[DEBUG] Generating system memory points: {event.get('event_name', 'unknown')}")
    
    system_memory_points = []
    
    # Get character name
    persona_name = "Zhu Botao"  # Default value
    if current_profile:
        # Check if current_profile contains profile field or is itself the profile structure
        if "profile" in current_profile:
            # current_profile contains profile field
            profile = current_profile["profile"]
        else:
            # current_profile itself is the profile structure
            profile = current_profile
        
        if "fixed" in profile:
            basic_info = profile["fixed"].get("basic_info", {})
            persona_name = basic_info.get("name", "Zhu Botao")
    
    # Process dynamic_updates for career events
    if event.get("event_type") == "career_event" and "dynamic_updates" in event:
        for update in event["dynamic_updates"]:
            type_to_update = update.get("type_to_update", "")
            update_direction = update.get("update_direction", "")
            before_dynamic = update.get("before_dynamic", {})
            after_dynamic = update.get("after_dynamic", {})
            changed_keys = update.get("changed_keys", [])
            
            # Generate system memory points for each changed key
            for key in changed_keys:
                if key in after_dynamic:
                    # Generate different memory point formats based on update type
                    after_value = after_dynamic[key]
                    
                    # Check type of after_value
                    if isinstance(after_value, dict):
                        if type_to_update == "social_relationships":
                            # Social relationship type
                            relationship_info = after_value
                            relationship_type = relationship_info.get("relationship_type", "")
                            description = relationship_info.get("description", "")
                            memory_content = f"{persona_name}'s social relationships increased {key}, {description}"
                        elif type_to_update == "work_experience":
                            # Work experience type
                            work_info = after_value
                            job_title = work_info.get("job_title", "")
                            company = work_info.get("company", "")
                            description = work_info.get("description", "")
                            memory_content = f"{persona_name}'s work experience increased {key}, {description}"
                        elif type_to_update == "skills":
                            # Skills type
                            skill_info = after_value
                            skill_level = skill_info.get("skill_level", "")
                            description = skill_info.get("description", "")
                            memory_content = f"{persona_name}'s skills increased {key}, {description}"
                        elif type_to_update == "achievements":
                            # Achievements type
                            achievement_info = after_value
                            achievement_type = achievement_info.get("achievement_type", "")
                            description = achievement_info.get("description", "")
                            memory_content = f"{persona_name}'s achievements increased {key}, {description}"
                        elif type_to_update == "health_status":
                            # Health status type - special handling
                            before_value = before_dynamic.get(key, "")
                            after_value_str = str(after_value)
                            
                            # Generate different memory points based on health indicators
                            if key == "physical_health":
                                memory_content = f"{persona_name}'s physical health status changed from \"{before_value}\" to \"{after_value_str}\""
                            elif key == "mental_health":
                                memory_content = f"{persona_name}'s mental health status changed from \"{before_value}\" to \"{after_value_str}\""
                            elif key == "physical_chronic_conditions":
                                if before_value == "" and after_value_str != "":
                                    memory_content = f"{persona_name} was diagnosed with {after_value_str}"
                                elif before_value != "" and after_value_str != "":
                                    memory_content = f"{persona_name}'s chronic disease changed from \"{before_value}\" to \"{after_value_str}\""
                                else:
                                    memory_content = f"{persona_name}'s chronic disease situation updated to \"{after_value_str}\""
                            elif key == "mental_chronic_conditions":
                                if before_value == "" and after_value_str != "":
                                    memory_content = f"{persona_name} was diagnosed with {after_value_str}"
                                elif before_value != "" and after_value_str != "":
                                    memory_content = f"{persona_name}'s mental illness changed from \"{before_value}\" to \"{after_value_str}\""
                                else:
                                    memory_content = f"{persona_name}'s mental illness situation updated to \"{after_value_str}\""
                            elif key == "situation_reason":
                                memory_content = f"{persona_name}'s health status changed, reason: {after_value_str}"
                            else:
                                memory_content = f"{persona_name}'s health status updated {key} from \"{before_value}\" to \"{after_value_str}\""
                        else:
                            # Other types, use generic format
                            description = str(after_value)
                            memory_content = f"{persona_name}'s {type_to_update} increased {key}, {description}"
                    else:
                        # If after_value is not a dictionary, need to get before_value for comparison
                        before_value = before_dynamic.get(key, "")
                        after_value_str = str(after_value)
                        
                        # Generate different memory points based on update type
                        if type_to_update == "health_status":
                            # Special handling for health status
                            if key == "physical_health":
                                memory_content = f"{persona_name}'s physical health status changed from \"{before_value}\" to \"{after_value_str}\""
                            elif key == "mental_health":
                                memory_content = f"{persona_name}'s mental health status changed from \"{before_value}\" to \"{after_value_str}\""
                            elif key == "physical_chronic_conditions":
                                if before_value == "" and after_value_str != "":
                                    memory_content = f"{persona_name} was diagnosed with {after_value_str}"
                                else:
                                    memory_content = f"{persona_name}'s chronic disease situation updated to \"{after_value_str}\""
                            elif key == "mental_chronic_conditions":
                                if before_value == "" and after_value_str != "":
                                    memory_content = f"{persona_name} was diagnosed with {after_value_str}"
                                else:
                                    memory_content = f"{persona_name}'s mental illness situation updated to \"{after_value_str}\""
                            elif key == "situation_reason":
                                memory_content = f"{persona_name}'s health status changed, reason: {after_value_str}"
                            else:
                                memory_content = f"{persona_name}'s health status updated {key} from \"{before_value}\" to \"{after_value_str}\""
                        else:
                            # Generic handling for other types
                            if before_value != "":
                                memory_content = f"{persona_name}'s {type_to_update} updated {key} from \"{before_value}\" to \"{after_value_str}\""
                            else:
                                memory_content = f"{persona_name}'s {type_to_update} updated {key} to \"{after_value_str}\""
                    
                    # Create system memory point
                    system_memory = {
                        "index": len(system_memory_points) + 1,
                        "memory_content": memory_content,
                        "memory_type": "Persona Memory",
                        "is_update": "True",
                        "original_memories": [],
                        "memory_source": "system",
                        "importance": config['memory_importance_weights']['system_memory'],
                        "timestamp": require_end_time_point(end_time_point)
                    }
                    
                    # Use UpdateTracker to fill original_memories
                    system_memory = UpdateTracker.track_system_memory_update(
                        system_memory, before_dynamic, [key], persona_name, type_to_update
                    )
                    
                    system_memory_points.append(system_memory)
    
    # Process daily routine event preference updates - fix to handle flattened preference structure
    elif event.get("event_type") == "daily_routine" and "before_preference" in event and "after_preference" in event:
        before_preference = event["before_preference"]
        after_preference = event["after_preference"]
        update_direction = event.get("update_direction", "")
        preference_type = event.get("preference_type", "")
        changed_index = event.get("changed_index", None)  # Get changed_index field
        
        # Handle deleted preferences
        if update_direction == "delete":
            # If changed_index exists, use it directly
            if changed_index is not None and isinstance(changed_index, (int, str)):
                try:
                    changed_index = int(changed_index)
                    if 0 <= changed_index < len(before_preference.get("memory_points", [])):
                        memory_point = before_preference["memory_points"][changed_index]
                        if memory_point.get("deleted"):
                            # Generate different memory point content based on preference type
                            if preference_type == "Clothing Preference":
                                memory_content = f"{persona_name} decided to reduce dependency on {memory_point['specific_item']}, choosing clothing more suitable for multiple occasions."
                            elif preference_type == "Game Preference":
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            elif preference_type == "Reading Preference":
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            elif preference_type == "Sports Preference":
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            elif preference_type == "Music Preference":
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            elif preference_type == "Food Preference":
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            else:
                                # Generic handling
                                memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                            
                            # Create system memory point
                            system_memory = {
                                "index": len(system_memory_points) + 1,
                                "memory_content": memory_content,
                                "memory_type": "Persona Memory",
                                "is_update": "True",
                                "original_memories": [f"{memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"],
                                "memory_source": "system",
                                "importance": config['memory_importance_weights']['system_memory'],
                                "timestamp": require_end_time_point(end_time_point)
                            }
                            
                            system_memory_points.append(system_memory)
                except (ValueError, IndexError):
                    print(f"[WARNING] changed_index {changed_index} is invalid, falling back to original logic")
            
            # If changed_index is None or invalid, use original logic
            if not system_memory_points or len(system_memory_points) == 0:
                for memory_point in before_preference.get("memory_points", []):
                    if memory_point.get("deleted"):
                        # Generate different memory point content based on preference type
                        if preference_type == "Clothing Preference":
                            memory_content = f"{persona_name} decided to reduce dependency on {memory_point['specific_item']}, choosing clothing more suitable for multiple occasions."
                        elif preference_type == "Game Preference":
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        elif preference_type == "Reading Preference":
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        elif preference_type == "Sports Preference":
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        elif preference_type == "Music Preference":
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        elif preference_type == "Food Preference":
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        else:
                            # Generic handling
                            memory_content = f"{persona_name} no longer prefers {memory_point['type_description']}: {memory_point['specific_item']}"
                        
                        # Create system memory point
                        system_memory = {
                            "index": len(system_memory_points) + 1,
                            "memory_content": memory_content,
                            "memory_type": "Persona Memory",
                            "is_update": "True",
                            "original_memories": [f"{memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"],
                            "memory_source": "system",
                            "importance": config['memory_importance_weights']['system_memory'],
                            "timestamp": require_end_time_point(end_time_point)
                        }
                        
                        system_memory_points.append(system_memory)
        
        # Handle modified preferences
        elif update_direction == "modify":
            # If changed_index exists, use it directly
            if changed_index is not None and isinstance(changed_index, (int, str)):
                try:
                    changed_index = int(changed_index)
                    before_points = before_preference.get("memory_points", [])
                    after_points = after_preference.get("memory_points", [])
                    
                    if 0 <= changed_index < len(before_points) and 0 <= changed_index < len(after_points):
                        before_point = before_points[changed_index]
                        after_point = after_points[changed_index]
                        
                        if before_point["type"] != after_point["type"] or before_point["reason"] != after_point["reason"]:
                            # Generate different memory point content based on preference type
                            if preference_type == "Clothing Preference":
                                memory_content = f"{persona_name} modified clothing preference from {after_point['specific_item']} to {after_point['specific_item']}."
                            elif preference_type == "Game Preference":
                                memory_content = f"{persona_name} gradually likes {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Reading Preference":
                                memory_content = f"{persona_name} starts to like {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Sports Preference":
                                memory_content = f"{persona_name} starts to prefer {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Music Preference":
                                memory_content = f"{persona_name} starts to like {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Food Preference":
                                memory_content = f"{persona_name} starts to prefer {after_point['specific_item']}, {after_point['reason']}"
                            else:
                                # Generic handling
                                memory_content = f"{persona_name} preference modified to {after_point['type_description']}: {after_point['specific_item']}. {after_point['reason']}"
                            
                            # Create system memory point
                            system_memory = {
                                "index": len(system_memory_points) + 1,
                                "memory_content": memory_content,
                                "memory_type": "Persona Memory",
                                "is_update": "True",
                                "original_memories": [f"{before_point['type_description']}: {before_point['specific_item']}. {before_point['reason']}"],
                                "memory_source": "system",
                                "importance": config['memory_importance_weights']['system_memory'],
                                "timestamp": require_end_time_point(end_time_point)
                            }
                            
                            system_memory_points.append(system_memory)
                except (ValueError, IndexError):
                    print(f"[WARNING] changed_index {changed_index} is invalid, falling back to original logic")
            
            # If changed_index is None or invalid, use original logic
            if not system_memory_points or len(system_memory_points) == 0:
                before_points = {mp["specific_item"]: mp for mp in before_preference.get("memory_points", [])}
                after_points = {mp["specific_item"]: mp for mp in after_preference.get("memory_points", [])}
                
                for item_name, after_point in after_points.items():
                    if item_name in before_points:
                        before_point = before_points[item_name]
                        if before_point["type"] != after_point["type"] or before_point["reason"] != after_point["reason"]:
                            # Generate different memory point content based on preference type
                            if preference_type == "Clothing Preference":
                                memory_content = f"{persona_name} modified clothing preference from {after_point['specific_item']} to {after_point['specific_item']}."
                            elif preference_type == "Game Preference":
                                memory_content = f"{persona_name} gradually likes {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Reading Preference":
                                memory_content = f"{persona_name} starts to like {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Sports Preference":
                                memory_content = f"{persona_name} starts to prefer {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Music Preference":
                                memory_content = f"{persona_name} starts to like {after_point['specific_item']}, {after_point['reason']}"
                            elif preference_type == "Food Preference":
                                memory_content = f"{persona_name} starts to prefer {after_point['specific_item']}, {after_point['reason']}"
                            else:
                                # Generic handling
                                memory_content = f"{persona_name} preference modified to {after_point['type_description']}: {after_point['specific_item']}. {after_point['reason']}"
                            
                            # Create system memory point
                            system_memory = {
                                "index": len(system_memory_points) + 1,
                                "memory_content": memory_content,
                                "memory_type": "Persona Memory",
                                "is_update": "True",
                                "original_memories": [f"{before_point['type_description']}: {before_point['specific_item']}. {before_point['reason']}"],
                                "memory_source": "system",
                                "importance": config['memory_importance_weights']['system_memory'],
                                "timestamp": require_end_time_point(end_time_point)
                            }
                            
                            system_memory_points.append(system_memory)
        
        # Handle added preferences
        elif update_direction == "add":
            # For add operations, changed_index should be null, directly process new memory points
            for memory_point in after_preference.get("memory_points", []):
                # Generate different memory point content based on preference type
                if preference_type == "Game Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                elif preference_type == "Clothing Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                elif preference_type == "Reading Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                elif preference_type == "Sports Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                elif preference_type == "Music Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                elif preference_type == "Food Preference":
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                else:
                    # Generic handling
                    memory_content = f"{persona_name}'s preference increased {memory_point['type_description']}: {memory_point['specific_item']}. {memory_point['reason']}"
                
                # Create system memory point
                system_memory = {
                    "index": len(system_memory_points) + 1,
                    "memory_content": memory_content,
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": [],
                    "memory_source": "system",
                    "importance": 1.0,  # System memory points importance fixed at 1.0
                    "timestamp": require_end_time_point(end_time_point)
                }
                
                system_memory_points.append(system_memory)
    
    print(f"[DEBUG] Generated {len(system_memory_points)} system memory points")
    return system_memory_points


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_memory_points_for_event(
    persona_info: str,
    current_events: List[str],
    stored_memory_points: List[str],
    event_info: Dict,
    user_motivation_type_info: str,
    memory_num: int,
    system_memory_points: List[Dict] = None,
    current_profile: Dict = None,
    previous_cost: Dict = None,
    start_time_point: str = None,
    end_time_point: str = None
) -> Dict:
    """Generate memory points for a single event"""
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set - this is required for memory point generation")
    
    # Build user input content
    event_info_str = json.dumps(event_info, ensure_ascii=False, indent=2)
    
    # Add current profile status information
    current_profile_info = ""
    if current_profile:
        try:
            current_profile_info = extract_current_profile_info(current_profile)
        except Exception as e:
            print(f"[WARNING] Failed to extract current profile information: {e}:{traceback.format_exc()}")
            current_profile_info = persona_info
    
    # Add system memory point information
    system_memory_info = ""
    if system_memory_points:
        system_memory_info = f"""
System memory points (already occupied {len(system_memory_points)} memory point slots):
{chr(10).join([f"- {mp['memory_content']}" for mp in system_memory_points])}

Remaining secondary memory points to generate: {memory_num - len(system_memory_points)}

Note: Please avoid generating secondary memory points that are duplicates of the above system memory points. Secondary memory points should focus on background, motivation, and impact of the event, rather than direct state updates.
"""
    else:
        system_memory_info = f"""
System memory points: None
Remaining secondary memory points to generate: {memory_num}
"""
    
    user_content = f"""
Character information: {current_profile_info if current_profile_info else persona_info}

Historical event list:
{chr(10).join(current_events)}

Stored memory points:
{chr(10).join(stored_memory_points)}

Current event information:
{event_info_str}

User motivation type: {user_motivation_type_info}

{system_memory_info}
"""
    
    try:
        print("[DEBUG] Sending memory points generation request to LLM...")

        # Find JSON part in thought chain output
        markers = ["Memory point generation result", "Generated memory points", "JSON result", "Final JSON", "Full JSON", "Generation result"]

        raw_response, cost_info = llm_request(stage4_1_events2memories_prompt, user_content, json_markers=markers)
        
        token_cost = calculate_cumulative_cost(previous_cost, cost_info)
        
        current_cost = cost_info
        cumulative_cost = token_cost.get('cumulative', {})
        print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
              f"Output: {current_cost.get('output_tokens', 'N/A')}, "
              f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
        print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
              f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        content = raw_response
        
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
            memory_result = json.loads(json_content)
            
            # Merge system memory points and secondary memory points
            if system_memory_points:
                # Get content of system memory points for deduplication
                system_memory_contents = [mp["memory_content"] for mp in system_memory_points]
                
                # Re-number and filter out duplicate content for secondary memory points
                secondary_memory_points = memory_result.get("memory_points", [])
                filtered_secondary_points = []
                
                for i, mp in enumerate(secondary_memory_points):
                    # Check if it's a duplicate of system memory points
                    if mp["memory_content"] not in system_memory_contents:
                        mp["index"] = len(system_memory_points) + len(filtered_secondary_points) + 1
                        mp["memory_source"] = "secondary"
                        # Ensure secondary memory points have importance field
                        if "importance" not in mp:
                            mp["importance"] = config['memory_importance_weights']['secondary_memory']
                        filtered_secondary_points.append(mp)
                
                # Merge memory points
                combined_memory_points = system_memory_points + filtered_secondary_points
                memory_result["memory_points"] = combined_memory_points
            
            # 添加成本信息到结果中
            memory_result["token_cost"] = token_cost
            return memory_result
        except json.JSONDecodeError as e:
            # Try to extract JSON using regex
            import re
            json_pattern = r'({[\s\S]*})'
            match = re.search(json_pattern, json_content)
            if match:
                try:
                    potential_json = match.group(1)
                    memory_result = json.loads(potential_json)
                    
                    # Merge system memory points and secondary memory points
                    if system_memory_points:
                        # Get content of system memory points for deduplication
                        system_memory_contents = [mp["memory_content"] for mp in system_memory_points]
                        
                        # Re-number and filter out duplicate content for secondary memory points
                        secondary_memory_points = memory_result.get("memory_points", [])
                        filtered_secondary_points = []
                        
                        for i, mp in enumerate(secondary_memory_points):
                            # Check if it's a duplicate of system memory points
                            if mp["memory_content"] not in system_memory_contents:
                                mp["index"] = len(system_memory_points) + len(filtered_secondary_points) + 1
                                mp["memory_source"] = "secondary"
                                # Ensure secondary memory points have importance field
                                if "importance" not in mp:
                                    mp["importance"] = config['memory_importance_weights']['secondary_memory']
                                filtered_secondary_points.append(mp)
                        
                        # Merge memory points
                        combined_memory_points = system_memory_points + filtered_secondary_points
                        memory_result["memory_points"] = combined_memory_points
                    
                    memory_result["token_cost"] = token_cost
                    return memory_result
                except json.JSONDecodeError:
                    pass
            
            # JSON parsing failed, raise error instead of returning default values
            raise ValueError("JSON parsing failed - this indicates a serious generation or parsing issue")
        
    except Exception as e:
        print(f"[ERROR] Error calling large language model to generate memory points: {e}:{traceback.format_exc()}")
        empty_cost = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "total_cost_usd": 0, "pricing_available": False
        }
        token_cost = calculate_cumulative_cost(previous_cost, empty_cost)
        raise ValueError(f"Error calling large language model to generate memory points: {e}:{traceback.format_exc()}")


def process_single_persona(persona_data: Dict, user_data: Dict) -> Dict:
    """Process single persona data, generate memory points"""
    # Extract character information
    try:
        persona_info = extract_profile_info(persona_data)
    except Exception as e:
        print(f"[ERROR] Failed to extract character information: {e}:{traceback.format_exc()}")
        raise ValueError(f"Cannot process user data, character information extraction failed: {e}:{traceback.format_exc()}")
    
    previous_cost = user_data.get('token_cost')
    
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }
    
    # Initialize memory points and event list
    current_memory_points = []
    current_events = []
    
    events = user_data['event_list']
    
    new_data = copy.deepcopy(user_data)
    new_data["event_list"] = []
    new_data["persona_info"] = persona_info
    
    # Initialize memory_points_all list, for summarizing memory points from all events
    memory_points_all = []
    
    # Generate utils results for internal use (not saved to separate files)
    utils_result = generate_utils_results(persona_data, user_data['uuid'])
    if utils_result:
        new_data["utils_results"] = utils_result
    
    # Initialize current profile state for tracking updates
    current_profile = copy.deepcopy(persona_data)
    
    # Process each event
    for event_index, event in enumerate(events):
        event_data = copy.deepcopy(event)
        
        # Convert event_time to dialogue time, require valid value
        event_time = event.get('event_time') or event.get('event_start_time') or event.get('event_end_time')
        if not isinstance(event_time, str) or not event_time.strip():
            raise ValueError(f"Event missing required time field: {event}")
        # start_time_point, end_time_point = convert_event_time_to_dialogue_time(event_time)
        
        # Process career event
        if event["event_type"] == "career_event":
            event["event_description"] = f"{event['stage_description']} {event['stage_result']}"
            # Remove unnecessary fields
            for field in ['stage_name', 'event_start_time', 'event_end_time', 'user_age', 'stage_description', 'stage_result', 'event_result']:
                event.pop(field, None)
        
        # Select user motivation type and generate system memory points
        if event["event_type"] in ["career_event", "daily_routine"]:
            user_motivation_type = random.choice(config['user_motivation_types'])
            user_motivation_type_info = f"{user_motivation_type['name']} -- {user_motivation_type['definition']}"
            memory_num = random.randint(
                config['memory_count_range'][0], 
                config['memory_count_range'][1]
            )

            start_time_point, end_time_point = convert_event_time_to_dialogue_time(event_time, memory_num)
            
            # Generate system memory points
            system_memory_points = generate_system_memory_points(event, current_profile, end_time_point)
            
            # Generate secondary memory points
            memory_result = generate_memory_points_for_event(
                persona_info,
                current_events,
                current_memory_points,
                event,
                user_motivation_type_info,
                memory_num,
                system_memory_points,
                current_profile,
                previous_cost,
                start_time_point,
                end_time_point
            )
        else:
            # Process initial event - use ConversationConverter class method from utils.py
            converter = ConversationConverter()
            if "initial_fixed" in event:
                memory_result = converter.generate_init_event_memories(persona_data, "initial_fixed")
            elif "initial_dynamic" in event:
                memory_result = converter.generate_init_event_memories(persona_data, "initial_dynamic")
            elif "initial_preference" in event:
                memory_result = converter.generate_init_event_memories(persona_data, "initial_preference")
            else:
                # Unknown event type - this should not happen
                raise ValueError(f"Unknown event type: {event.get('event_type', 'unknown')}. Only 'initial_fixed', 'initial_dynamic', 'initial_preference', 'career_event', and 'daily_routine' are supported.")
            
            memory_num = memory_result.get("memory_points_count")
            start_time_point, end_time_point = convert_event_time_to_dialogue_time(event_time, memory_num)
            
            # Override time points for init events to ensure consistency
            if memory_result:
                memory_result["start_time_point"] = start_time_point
                memory_result["end_time_point"] = end_time_point
            
            # For init events, directly use the generated results
            if "initial_fixed" in event or "initial_dynamic" in event or "initial_preference" in event:
                # Add timestamp field and event source number to each memory point
                for memory_point in memory_result.get('memory_points', []):
                    memory_point['timestamp'] = end_time_point  # Use end_time_point for all memory points
                    memory_point['event_source'] = event_index  # Add event source number
                    
                    # Ensure all memory points have importance field
                    if 'importance' not in memory_point:
                        memory_point['importance'] = config['memory_importance_weights']['init_event_memory']
                
                # Add to event data
                event_data["dialogue_info"] = memory_result
                
                # Add memory points to memory_points_all list
                memory_points_all.extend(memory_result.get('memory_points', []))
                
                # Update current memory points and event list
                current_memory_points.extend(
                    f"  - [{memory_result['start_time_point']}][{i['memory_type']}]{i['memory_content']}" 
                    for i in memory_result.get('memory_points', [])
                )
                
                new_data["event_list"].append(event_data)
                
                # Format event time - use hyphen instead of colon
                try:
                    event_time = datetime.strptime(event['event_time'], "%Y-%m-%d").strftime("%b %d, %Y")
                    current_events.append(
                        f"  - [{event_time}]{event['event_name']} - {event['event_description']}"
                    )
                except Exception as e:
                    print(f"[ERROR] Event time formatting failed: {e}:{traceback.format_exc()}")
                    current_events.append(
                        f"  - [Unknown Date]{event['event_name']} - {event['event_description']}"
                    )
                continue
        
        # Ensure memory_result has correct time points
        memory_result['start_time_point'] = start_time_point
        memory_result['end_time_point'] = end_time_point
        
        # Add timestamp field and event source number to all memory points
        for memory_point in memory_result.get('memory_points', []):
            memory_point['timestamp'] = end_time_point  # Use end_time_point for all memory points
            memory_point['event_source'] = event_index  # Add event source number
            
            # Ensure all memory points have memory_source field
            if 'memory_source' not in memory_point:
                # If memory point does not have memory_source field, it's a secondary memory point
                memory_point['memory_source'] = 'secondary'
            
            # Ensure all memory points have importance field
            if 'importance' not in memory_point:
                if memory_point.get('memory_source') == 'system':
                    memory_point['importance'] = config['memory_importance_weights']['system_memory']
                else:
                    memory_point['importance'] = config['memory_importance_weights']['secondary_memory']
        
        # Update memory point result
        memory_result["memory_points_count"] = len(memory_result.get("memory_points", []))
        memory_result["user_motivation_type"] = user_motivation_type_info
        
        # 累加当前事件的 LLM 调用成本
        if "token_cost" in memory_result and memory_result["token_cost"]:
            current_cost = memory_result["token_cost"].get("current_stage", {})
            current_stage_total_cost["input_tokens"] += current_cost.get("input_tokens", 0)
            current_stage_total_cost["output_tokens"] += current_cost.get("output_tokens", 0)
            current_stage_total_cost["total_tokens"] += current_cost.get("total_tokens", 0)
            current_stage_total_cost["total_cost_usd"] += current_cost.get("total_cost_usd", 0.0)
            if current_stage_total_cost["model"] is None:
                current_stage_total_cost["model"] = current_cost.get("model")
            current_stage_total_cost["pricing_available"] = current_cost.get("pricing_available", False)
        
        # Add to event data
        event_data["dialogue_info"] = memory_result
        
        # Add memory points to memory_points_all list
        memory_points_all.extend(memory_result.get('memory_points', []))
        
        # Update current memory points and event list
        current_memory_points.extend(
            f"  - [{memory_result['start_time_point']}][{i['memory_type']}]{i['memory_content']}" 
            for i in memory_result.get('memory_points', [])
        )
        
        new_data["event_list"].append(event_data)
        
        # Format event time - require valid value
        event_time_raw = event.get('event_time') or event.get('event_start_time') or event.get('event_end_time')
        if not isinstance(event_time_raw, str) or not event_time_raw.strip():
            raise ValueError(f"Event missing required time field for formatting: {event}")
        try:
            event_time = datetime.strptime(event_time_raw.strip(), "%Y-%m-%d").strftime("%b %d, %Y")
        except Exception as exc:
            raise ValueError(f"Invalid event_time format for event list entry: {event_time_raw}: {exc}") from exc

        current_events.append(
            f"  - [{event_time}]{event['event_name']} - {event['event_description']}"
        )
        
        # Apply event updates to current profile
        if event.get("event_type") == "career_event" and "dynamic_updates" in event:
            # Apply career event dynamic updates
            current_profile = ProfileUpdater.apply_dynamic_updates(current_profile, event["dynamic_updates"])
        elif event.get("event_type") == "daily_routine":
            # Apply daily routine preference updates
            current_profile = ProfileUpdater.apply_preference_updates(current_profile, event)
        
        print(f"[DEBUG] Event {event_index + 1} processed, profile updated")
    
    # Add memory_points_all field to result
    new_data["memory_points_all"] = memory_points_all
    
    final_token_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
    
    new_data["token_cost"] = final_token_cost
    
    return new_data


def process_all_users(regenerate: bool = True):
    """Process all user data"""
    input_file = config['file_paths']['input_file']
    output_file = config['file_paths']['output_file']
    
    print(f"Batch processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Regenerate mode: {regenerate}")
    
    # If in regenerate mode, delete previous output file
    if regenerate and os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[DEBUG] Deleted previous output file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Error deleting output file: {e}:{traceback.format_exc()}")
    
    # If not in regenerate mode, check if output file exists and contains data
    if not regenerate and os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                existing_count = sum(1 for _ in reader)
            if existing_count > 0:
                print(f"[DEBUG] Output file already exists and contains {existing_count} data, skipping processing")
                return True
        except Exception as e:
            print(f"[ERROR] Error checking output file: {e}:{traceback.format_exc()}")
    
    try:
        all_user_data = []
        with jsonlines.open(input_file) as reader:
            for user_data in reader:
                user_uuid = user_data.get('uuid')
                if not user_uuid:
                    continue
                
                # Extract persona information directly from user_data
                persona_data = user_data['profile']
                
                # Ensure persona_data contains necessary field structure
                if not isinstance(persona_data, dict):
                    print(f"[ERROR] persona_data is not a dictionary type: {type(persona_data)}")
                    continue
                    
                # Check if preferences field exists and is in correct format
                if 'preferences' in persona_data:
                    preferences = persona_data['preferences']
                    if isinstance(preferences, dict):
                        for pref_type, pref_info in preferences.items():
                            if not isinstance(pref_info, dict) or 'memory_points' not in pref_info:
                                print(f"[ERROR] Fixing preferences data structure: {pref_type}")
                                # If pref_info is not a dictionary or missing memory_points, set to empty list
                                if not isinstance(pref_info, dict):
                                    persona_data['preferences'][pref_type] = {'memory_points': [], 'overall_description': ''}
                                elif 'memory_points' not in pref_info:
                                    persona_data['preferences'][pref_type]['memory_points'] = []
                
                all_user_data.append(user_data)
        
        def process_single_user_data(user_data):
            user_uuid = user_data.get('uuid')
            try:
                # Extract persona information directly from user_data
                persona_data = user_data['profile']
                
                # Process user data
                result_data = process_single_persona(persona_data, user_data)
                print(f"[DEBUG] Successfully processed user {user_uuid}")
                return result_data
            except Exception as e:
                print(f"[ERROR] Error processing user {user_uuid}: {e}:{traceback.format_exc()}")
                return None
        
        processed_results = parallel_process(all_user_data, process_single_user_data, "Stage4.1-Events2Memories")
        
        valid_results = [result for result in processed_results if result is not None]
        with jsonlines.open(output_file, 'w') as writer:
            for result in valid_results:
                writer.write(result)
        
        print(f"[DEBUG] Batch processing completed, processed {len(valid_results)} user data")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during batch processing: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate memory points for persona information')
    parser.add_argument('--regenerate', action='store_true', default=True, 
                       help='Whether to regenerate completely (default: True)')
    parser.add_argument('--skip-existing', action='store_true', 
                       help='Skip existing parts (mutually exclusive with --regenerate)')
    
    args = parser.parse_args()
    
    # Determine generation mode
    if args.skip_existing:
        regenerate = False
    else:
        regenerate = args.regenerate
    
    print(f"Generation mode: {'Regenerate' if regenerate else 'Skip existing'}")
    process_all_users(regenerate)
