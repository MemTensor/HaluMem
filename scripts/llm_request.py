import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)
from openai import OpenAI


log_file = 'llm_caller.log'
if os.path.exists(log_file):
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('') 

logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MODEL_PRICING = {
    'gpt-4o': {
        'input': 2.50,
        'output': 10.00
    },
    'gpt-4o-2024-05-13': {
        'input': 5.00,
        'output': 15.00
    },
    'gpt-4o-2024-08-06': {
        'input': 2.50,
        'output': 10.00
    },
    'gpt-4o-2024-11-20': {
        'input': 2.50,
        'output': 10.00
    },
    'gpt-4o-mini': {
        'input': 0.15,
        'output': 0.60
    }
}


def calculate_cumulative_cost(previous_cost: Optional[Dict], current_cost: Dict) -> Dict:
    
    result = {
        "current_stage": current_cost,
        "cumulative": {
            "input_tokens": current_cost.get("input_tokens", 0) or 0,
            "output_tokens": current_cost.get("output_tokens", 0) or 0,
            "total_tokens": current_cost.get("total_tokens", 0) or 0,
            "total_cost_usd": current_cost.get("total_cost_usd", 0) or 0
        }
    }
    
    if previous_cost and isinstance(previous_cost, dict):

        if "cumulative" in previous_cost:
            prev_cumulative = previous_cost["cumulative"]
        elif "current_stage" in previous_cost:
            prev_cumulative = previous_cost["current_stage"]
        else:
            prev_cumulative = previous_cost
        
        if prev_cumulative:
            result["cumulative"]["input_tokens"] += prev_cumulative.get("input_tokens", 0) or 0
            result["cumulative"]["output_tokens"] += prev_cumulative.get("output_tokens", 0) or 0
            result["cumulative"]["total_tokens"] += prev_cumulative.get("total_tokens", 0) or 0
            result["cumulative"]["total_cost_usd"] += prev_cumulative.get("total_cost_usd", 0) or 0
    
    if result["cumulative"]["total_cost_usd"]:
        result["cumulative"]["total_cost_usd"] = round(result["cumulative"]["total_cost_usd"], 6)
    
    return result


def _extract_json_from_content(content: str, markers: List[str]) -> Dict:
    
    json_content = content
    
    for marker in markers:
        if marker in json_content:
            parts = json_content.split(marker, 1)
            if len(parts) > 1:
                json_content = parts[1].strip()
                break
    
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
    
    if '{' in json_content and '}' in json_content:
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}') + 1
        if end_idx > start_idx:
            json_content = json_content[start_idx:end_idx].strip()
    
    try:
        parsed_json = json.loads(json_content)
        logger.debug("Successfully parsed JSON")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e}, trying regex extraction")
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, json_content)
        if match:
            try:
                potential_json = match.group(1)
                parsed_json = json.loads(potential_json)
                logger.debug("Successfully extracted JSON through regex")
                return parsed_json
            except json.JSONDecodeError:
                logger.warning("Content extracted through regex is not valid JSON")
        
        raise ValueError(f"Failed to parse JSON from content: {json_content[:200]}...")


def _calculate_cost(model: str, usage: Optional[Any]) -> Dict:

    if usage is None:
        logger.warning("No usage information available in API response")
        return {
            "input_tokens": None, "output_tokens": None, "total_tokens": None, "model": model,
            "input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None,
            "pricing_available": False, "note": "Usage information not available"
        }
    
    usage_dict = usage.model_dump()
    input_tokens = usage_dict.get('prompt_tokens', 0)
    output_tokens = usage_dict.get('completion_tokens', 0)
    
    cost_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model": model
    }
    
    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        cost_info.update({
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "pricing_available": True
        })
    else:
        cost_info.update({
            "input_cost_usd": None, "output_cost_usd": None, "total_cost_usd": None,
            "pricing_available": False, "note": f"Pricing not available for model: {model}"
        })
    
    return cost_info


load_dotenv()

RETRY_TIMES = int(os.getenv('RETRY_TIMES'))
WAIT_TIME_LOWER = int(os.getenv('WAIT_TIME_LOWER'))
WAIT_TIME_UPPER = int(os.getenv('WAIT_TIME_UPPER'))

client = OpenAI(
    base_url=os.getenv('OPENAI_BASE_URL'),
    api_key=os.getenv('OPENAI_API_KEY')
)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_request(
    system_prompt: str, 
    user_prompt: str, 
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    timeout: int = 300,
    return_parsed_json: bool = False,
    extract_json: bool = True,
    json_markers: Optional[List[str]] = None
) -> tuple:
    
    final_model = model or os.getenv('OPENAI_MODEL')
    if not final_model:
        raise ValueError("Model name must be provided either as a parameter or in the OPENAI_MODEL environment variable.")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Build request parameters dynamically based on what's available
    request_params = {
        "model": final_model,
        "messages": messages
    }
    
    # Add optional parameters only if they exist in environment or are explicitly provided
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    elif os.getenv('OPENAI_MAX_TOKENS'):
        request_params["max_tokens"] = int(os.getenv('OPENAI_MAX_TOKENS'))
    
    if temperature is not None:
        request_params["temperature"] = temperature
    elif os.getenv('OPENAI_TEMPERATURE'):
        request_params["temperature"] = float(os.getenv('OPENAI_TEMPERATURE'))
    
    if timeout is not None:
        request_params["timeout"] = timeout
    elif os.getenv('OPENAI_TIMEOUT'):
        request_params["timeout"] = int(os.getenv('OPENAI_TIMEOUT'))
        
    response = client.chat.completions.create(**request_params)

    content = response.choices[0].message.content.strip()
    logger.debug(f"[DEBUG] API response content: {content[:200]}...")

    cost_info = _calculate_cost(final_model, response.usage)

    if not extract_json:
        return content, cost_info
    
    if json_markers is None:
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result",
            "Dialogue Generation Result", "Generated Dialogue", "JSON Result",
            "Generation Result"
        ]
    
    parsed_json = _extract_json_from_content(content, json_markers)

    if return_parsed_json:
        return parsed_json, cost_info
    else:
        return content, cost_info
