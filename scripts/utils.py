import json
import os
from typing import Dict, List, Any
from string import Template


class ConversationConverter:
    """Tool class for converting persona data to dialogue format"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize converter
        
        Args:
            prompts_dir: Directory containing template files
        """
        self.prompts_dir = prompts_dir
        self.templates = self._load_templates()
    
    def _normalize_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize data structure, compatible with two formats:
        1. Direct format: data['field'] = value
        2. Nested format: data['field']['init'] = value
        
        Args:
            data: Original data
            
        Returns:
            Standardized data
        """
        normalized = {}
        
        for key, value in data.items():
            if isinstance(value, dict) and 'init' in value:
                # Nested format: extract data from init
                normalized[key] = value['init']
            else:
                # Direct format: use directly
                normalized[key] = value
        
        return normalized
    
    def _normalize_terminal_punctuation(self, text: str) -> str:
        """Normalize ending punctuation: collapse repeats and ensure at most one terminal dot."""
        if not isinstance(text, str):
            return text
        import re
        s = text.strip()
        # Collapse multiple terminal punctuation marks
        s = re.sub(r"([.!?]){2,}$", r"\1", s)
        # Remove trailing spaces before punctuation
        s = re.sub(r"\s+([.!?])$", r"\1", s)
        return s

    def _sanitize_reason_text(self, reason: str) -> str:
        """Sanitize reason phrases to avoid duplicated lead like 'I like/I love/I dislike ...'."""
        if not isinstance(reason, str):
            return ""
        import re
        s = reason.strip()
        # Remove leading first-person sentiment phrases
        s = re.sub(r"^i\s+(do\s+not|don't)?\s*(like|love|enjoy|prefer|dislike|hate)\b[:,\-\s]*",
                   "", s, flags=re.IGNORECASE)
        # If starts with because, keep; else optionally add 'because '
        if not re.match(r"^(because\b|as\b|since\b)", s, flags=re.IGNORECASE):
            s = f"because {s}"
        # Normalize spaces and terminal punctuation
        s = re.sub(r"\s+", " ", s).strip()
        s = self._normalize_terminal_punctuation(s)
        # Ensure no trailing comma
        s = re.sub(r",$", "", s)
        return s

    def _normalize_person_references(self, text: str) -> str:
        """
        Convert third person references to first person in text
        
        Args:
            text: Original text
            
        Returns:
            Text with first person references
        """
        if not text:
            return text
        
        # More comprehensive person reference conversion
        replacements = {
            'she': 'I',
            'her': 'my', 
            'hers': 'mine',
            'herself': 'myself',
            'he': 'I',  # Just in case
            'him': 'me',
            'his': 'my',
            'himself': 'myself'
        }
        
        result = text
        for old, new in replacements.items():
            # Use word boundaries to avoid partial replacements
            import re
            result = re.sub(r'\b' + re.escape(old) + r'\b', new, result, flags=re.IGNORECASE)
        
        return result

    # =========================
    # Post-process text utilities
    # =========================
    def postprocess_text(self, text: str) -> str:
        """Run unified text cleanup after generation.
        Covers: duplicated phrases (I like I love/ I like I dislike), punctuation, a/an, industry phrases, residual labels, simple pronoun fix, etc.
        """
        if not isinstance(text, str) or not text.strip():
            return text
        import re
        s = text.strip()

        # Preserve ellipsis by temporary placeholder to avoid being collapsed
        ELLIPSIS_TOKEN = "<ELLIPSIS>"
        s = s.replace("...", ELLIPSIS_TOKEN)

        # 1) Basic whitespace and label cleanup
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\b(User|Assistant):\s*", "", s)

        # 2) Merge duplicate terminal punctuation and double dots
        # Collapse runs of periods (not ellipsis due to placeholder)
        s = re.sub(r"\.(\s*\.)+", ".", s)
        # Collapse repeated ! or ?
        s = re.sub(r"!{2,}", "!", s)
        s = re.sub(r"\?{2,}", "?", s)

        # 3) Fix missing period before connectors
        s = re.sub(r"\b(No chronic diseases)\s+(In terms of)\b", r"\1. \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(No mental health issues)\s+(I)\b", r"\1. \2", s, flags=re.IGNORECASE)
        # Lowercase Because after comma
        s = re.sub(r",\s+Because\b", ", because", s)

        # 4) Remove duplicate determiners/words
        s = re.sub(r"\bthe\s+the\b", "the", s, flags=re.IGNORECASE)
        s = re.sub(r"\bindustry\s+industry\b", "industry", s, flags=re.IGNORECASE)

        # 5) Industry phrase corrections
        s = re.sub(r"\bthe\s+various\s+industry\b", "various industries", s, flags=re.IGNORECASE)
        s = re.sub(r"\bvarious\s+industry\b", "various industries", s, flags=re.IGNORECASE)

        # 6) Preference phrase de-dup and normalization
        # Reduce repeated like/love/enjoy/appreciate/dislike/hate at sentence start
        def _normalize_preference_sentence(m: re.Match) -> str:
            lead = m.group(1)
            rest = m.group(2)
            # If contradictory phrase exists later (dislike/hate), prefer negative
            negative = re.search(r"\b(don't like|do not like|dislike|hate)\b", rest, flags=re.IGNORECASE) is not None
            # Strip repeated leading verbs from rest
            rest2 = re.sub(r"\b(i\s+(?:like|love|enjoy|appreciate|dislike|hate)\b\s*)+", "", rest, flags=re.IGNORECASE)
            # Normalize reason prefix
            if not re.search(r"\b(because|as|since)\b", rest2, flags=re.IGNORECASE):
                # insert 'because' before capitalized Because-less reason starting with because clause? If comma exists, keep as is.
                rest2 = re.sub(r"\s*,\s*", ", ", rest2)
                # If there is a comma then keep; else add 'because '
                if not re.search(r"\b(because|as|since)\b", rest2, flags=re.IGNORECASE):
                    rest2 = f"because {rest2}".strip()
            lead_norm = "I don't like" if negative else "I like"
            sent = f"{lead_norm} {rest2}"
            return sent

        s = re.sub(r"\b(I\s+(?:like|love|enjoy|appreciate|dislike|hate))\s+(.*?)(?=(?:[.!?]|$))",
                    _normalize_preference_sentence, s, flags=re.IGNORECASE)

        # 7) Article a/an heuristic
        # Handle some exceptions first
        s = re.sub(r"\b(an)\s+(university|European|euro|use|user)\b", r"a \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(a)\s+(hour|honest|honor|heir)\b", r"an \2", s, flags=re.IGNORECASE)
        # General vowel/consonant rules
        s = re.sub(r"\ban\s+([b-df-hj-np-tv-z])", r"a \1", s, flags=re.IGNORECASE)
        s = re.sub(r"\ba\s+([aeiou])", r"an \1", s, flags=re.IGNORECASE)

        # 8) Simple pronoun fix pattern: "X and I ... my advice ... to me" -> their advice
        s = re.sub(r"\band I[^.]*?\bmy advice\b([^.]*)\bto me\b", r"and I\1their advice to me", s, flags=re.IGNORECASE)

        # 9) Final punctuation normalization per sentence boundaries
        s = re.sub(r"\s*([.!?])\s*", r"\1 ", s)
        # Remove any accidental double punctuation remnants again
        s = re.sub(r"([.!?])\s*\1+", r"\1", s)
        s = s.strip()
        s = re.sub(r"\s+", " ", s).strip()

        # Restore ellipsis
        s = s.replace(ELLIPSIS_TOKEN, "...")

        return s

    def postprocess_message_list(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not isinstance(messages, list):
            return messages
        processed = []
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                new_msg = dict(msg)
                new_msg['content'] = self.postprocess_text(msg['content'])
                processed.append(new_msg)
            else:
                processed.append(msg)
        return processed

    def postprocess_conversations(self, conversations: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(conversations, dict):
            return conversations
        new_conv = dict(conversations)
        if 'fixed_conversation' in new_conv:
            new_conv['fixed_conversation'] = self.postprocess_message_list(new_conv['fixed_conversation'])
        if 'dynamic_conversation' in new_conv:
            new_conv['dynamic_conversation'] = self.postprocess_message_list(new_conv['dynamic_conversation'])
        if 'preferences_conversations' in new_conv and isinstance(new_conv['preferences_conversations'], list):
            new_pref = []
            for conv in new_conv['preferences_conversations']:
                new_pref.append(self.postprocess_message_list(conv))
            new_conv['preferences_conversations'] = new_pref
        return new_conv

    # =============
    # Public helpers for external pipelines (e.g., stage5)
    # =============
    def postprocess_structure(self, data: Any) -> Any:
        """Recursively apply postprocess_text to all string leaves in any nested structure.
        Safe and idempotent. Non-string leaves are returned as-is.
        """
        if isinstance(data, str):
            return self.postprocess_text(data)
        if isinstance(data, list):
            return [self.postprocess_structure(item) for item in data]
        if isinstance(data, dict):
            return {key: self.postprocess_structure(value) for key, value in data.items()}
        return data

    def postprocess_stage_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper to postprocess any stage JSON/JSONL payload (e.g., stage5 outputs).
        Applies text cleanup to all string fields within the payload.
        """
        if not isinstance(payload, dict):
            return payload
        return self.postprocess_structure(payload)
    
    def _load_templates(self) -> Dict[str, str]:
        """Load all template files"""
        templates = {}
        template_files = {
            'fixed': 'utils_fixed_conversation_template.txt',
            'dynamic': 'utils_dynamic_conversation_template.txt',
            'preferences': 'utils_preferences_conversation_template.txt'
        }
        
        for key, filename in template_files.items():
            filepath = os.path.join(self.prompts_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    templates[key] = f.read()
            else:
                raise FileNotFoundError(f"Template file does not exist: {filepath}")
        
        return templates
    
    def _parse_conversation_to_messages(self, conversation_text: str) -> List[Dict[str, str]]:
        """
        Parse dialogue text to standard LLM messages format
        
        Args:
            conversation_text: Dialogue text
            
        Returns:
            List of messages, each containing role and content
        """
        messages = []
        current_role = None
        current_content = []
        
        lines = conversation_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('User: '):
                # Save previous message
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new user message
                current_role = "user"
                current_content = [line[6:].strip()]
                
            elif line.startswith('Assistant: '):
                # Save previous message
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new assistant message
                current_role = "assistant"
                current_content = [line[11:].strip()]
                
            else:
                # Continue current message content
                if current_role:
                    current_content.append(line)
        
        # Save last message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })
        
        return messages
    
    def _format_fixed_conversation(self, fixed_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format dialogue for fixed section"""
        # Standardize data structure
        fixed_data = self._normalize_data_structure(fixed_data)
        
        # Check if fixed_data is empty or missing necessary fields
        if not fixed_data or 'family_life' not in fixed_data:
            print("[ERROR] fixed_data is empty or missing family_life field, cannot generate dialogue")
            raise ValueError("fixed_data is empty or missing family_life field, cannot generate dialogue")
        
        # Process family members
        family = fixed_data['family_life']
        
        # Parent status mapping
        parent_status_map = {
            'both_alive': 'Both parents are alive',
            'father_alive': 'Father is alive',
            'mother_alive': 'Mother is alive',
            'both_deceased': 'Both parents are deceased',
            'one_deceased': 'One parent is deceased'
        }
        parent_status = parent_status_map.get(family['parent_status'], family['parent_status'])
        
        # Partner status mapping
        partner_status_map = {
            'single': 'Single',
            'married': 'Married',
            'divorced': 'Divorced',
            'widowed': 'Widowed',
            'dating': 'Dating'
        }
        partner_status = partner_status_map.get(family['partner_status'], family['partner_status'])
        
        # Child status mapping
        child_status_map = {
            'no_children': 'No children',
            'one_child': 'One child',
            'two_children': 'Two children',
            'three_children': 'Three children'
        }
        child_status = child_status_map.get(family['child_status'], family['child_status'])
        
        # Parent member information
        parent_members_entries = []
        if 'parent_members' in family and isinstance(family['parent_members'], list):
            for member in family['parent_members']:
                if isinstance(member, dict) and 'member_type' in member and 'birth_date' in member and 'description' in member:
                    entry = f"My {member['member_type']} was born on {member['birth_date']}, {member['description']}."
                    parent_members_entries.append(entry)
                else:
                    print(f"[DEBUG] Skipping invalid parent member data: {member}")
        else:
            print(f"[DEBUG] parent_members is not a list or does not exist: {family.get('parent_members', 'not found')}")
        
        # Partner information
        partner_entry = ""
        if 'partner' in family and family['partner'] and isinstance(family['partner'], dict):
            partner = family['partner']
            if 'member_type' in partner and 'birth_date' in partner and 'description' in partner:
                partner_entry = f"My {partner['member_type']} was born on {partner['birth_date']}, {partner['description']}."
            else:
                print(f"[DEBUG] Partner data missing necessary fields: {partner}")
        else:
            print(f"[DEBUG] Partner data: {family}")
            print(f"[DEBUG] Partner data does not exist or format error: {family.get('partner', 'not found')}")
        
        # Child information
        child_members_entries = []
        if 'child_members' in family and isinstance(family['child_members'], list):
            for member in family['child_members']:
                if isinstance(member, dict) and 'member_type' in member and 'birth_date' in member and 'description' in member:
                    entry = f"My {member['member_type']} was born on {member['birth_date']}, {member['description']}."
                    child_members_entries.append(entry)
                else:
                    print(f"[DEBUG] Skipping invalid child member data: {member}")
        else:
            print(f"[DEBUG] child_members is not a list or does not exist: {family.get('child_members', 'not found')}")
        
        # Prepare template variables
        template_vars = {
            'name': fixed_data.get('basic_info', {}).get('name', 'Unknown'),
            'gender': fixed_data.get('basic_info', {}).get('gender', 'Unknown'),
            'birth_date': fixed_data.get('basic_info', {}).get('birth_date', 'Unknown'),
            'current_age': fixed_data.get('age', {}).get('current_age', 'Unknown'),
            'location': fixed_data.get('basic_info', {}).get('location', 'Unknown'),
            'highest_degree': fixed_data.get('education', {}).get('highest_degree', 'Unknown'),
            'major': fixed_data.get('education', {}).get('major', 'Unknown'),
            'mbti': fixed_data.get('personality', {}).get('mbti', 'Unknown'),
            'personality_tags': ', '.join(fixed_data.get('personality', {}).get('tags', [])),
            'parent_status': parent_status,
            'partner_status': partner_status,
            'child_status': child_status,
            'parent_members_entries': '\n'.join(parent_members_entries),
            'partner_entry': partner_entry,
            'child_members_entries': '\n'.join(child_members_entries),
            'family_description': family.get('family_description', ''),
            'statement': fixed_data.get('life_goal', {}).get('statement', ''),
            'motivation': fixed_data.get('life_goal', {}).get('motivation', ''),
            'target_metrics': fixed_data.get('life_goal', {}).get('target_metrics', '')
        }
        
        # Use template for formatting
        template = Template(self.templates['fixed'])
        conversation_text = template.safe_substitute(template_vars)
        
        # Parse to messages format
        messages = self._parse_conversation_to_messages(conversation_text)
        # Final postprocess to ensure cleanliness for callers like stage5
        messages = self.postprocess_message_list(messages)
        return messages
    
    def _format_dynamic_conversation(self, dynamic_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format dialogue for dynamic section"""
        # Standardize data structure
        dynamic_data = self._normalize_data_structure(dynamic_data)
        
        # Check if dynamic_data is empty
        if not dynamic_data:
            print("[ERROR] dynamic_data is empty, cannot generate dialogue")
            raise ValueError("dynamic_data is empty, cannot generate dialogue")
        
        # Process merged dynamic data structure
        # dynamic_data is now a dictionary containing multiple dynamic types
        # For example: {"career_status": {...}, "health_status": {...}, "social_relationships": {...}}
        
        # Extract data for each dynamic type
        career_status = dynamic_data.get('career_status', {})
        health_status = dynamic_data.get('health_status', {})
        social_relationships = dynamic_data.get('social_relationships', {})
        
        # Process employment status
        employment_status_map = {
            'employed': 'Employed',
            'unemployed': 'Unemployed',
            'student': 'Student',
            'retired': 'Retired'
        }
        employment_status = employment_status_map.get(career_status.get('employment_status', ''), career_status.get('employment_status', ''))
        
        # Process social relationships
        relationship_entries = []
        if isinstance(social_relationships, dict):
            for name, info in social_relationships.items():
                if isinstance(info, dict) and 'description' in info and 'relationship_type' in info:
                    # Use original description (input already in first person), avoid forced pronoun mapping
                    description = (info['description'] or '').strip()
                    entry = f"{name} is my {info['relationship_type']}, {description}"
                    relationship_entries.append(entry)
                else:
                    print(f"[DEBUG] Skipping invalid social relationship data: {name} -> {info}")
        else:
            print(f"[DEBUG] social_relationships is not a dictionary: {social_relationships}")
        
        # Prepare template variables
        template_vars = {
            'employment_status': employment_status,
            'industry': career_status.get('industry', ''),
            'company_name': career_status.get('company_name', ''),
            'job_title': career_status.get('job_title', ''),
            'monthly_income': career_status.get('monthly_income', ''),
            'savings_amount': career_status.get('savings_amount', ''),
            'career_description': career_status.get('career_description', ''),
            'physical_health': health_status.get('physical_health', ''),
            'physical_chronic_conditions': health_status.get('physical_chronic_conditions') or "No chronic diseases.",
            'mental_health': health_status.get('mental_health', ''),
            'mental_chronic_conditions': health_status.get('mental_chronic_conditions') or "No mental health issues.",
            'situation_reason': health_status.get('situation_reason', ''),
            'relationship_entries': '. '.join(relationship_entries) + '.'
        }
        
        # Use template for formatting
        template = Template(self.templates['dynamic'])
        conversation_text = template.safe_substitute(template_vars)
        
        print(f"[DEBUG] conversation_text: {conversation_text}")
        
        # Parse to messages format
        messages = self._parse_conversation_to_messages(conversation_text)
        # Final postprocess to ensure cleanliness for callers like stage5
        messages = self.postprocess_message_list(messages)
        return messages
    
    def _format_preferences_conversation(self, preferences_data: Dict[str, Any]) -> List[List[Dict[str, str]]]:
        """Format dialogue for preferences section, generate natural dialogue for each preference type"""
        # Check if preferences_data is empty
        if not preferences_data:
            print("[ERROR] preferences_data is empty, cannot generate dialogue")
            raise ValueError("preferences_data is empty, cannot generate dialogue")
        
        # Collect all dialogue message lists, one list for each preference
        all_conversations = []
        
        # Process merged preferences data structure
        # preferences_data is now a dictionary containing multiple preference types
        # For example: {"Movie Preference": {...}, "Travel Preference": {...}}
        
        for preference_type, preference_info in preferences_data.items():
            # Check if preference_info contains necessary fields
            if not preference_info:
                print(f"[DEBUG] preference_info is empty, skipping: {preference_type}")
                continue
                
            if 'memory_points' not in preference_info:
                print(f"[DEBUG] preference_info missing memory_points field, skipping: {preference_type}")
                continue
                
            # Process memory points, convert to natural dialogue language
            memory_points_text = []
            memory_points = preference_info.get('memory_points', [])
            
            if isinstance(memory_points, list):
                for memory_point in memory_points:
                    if isinstance(memory_point, dict) and 'specific_item' in memory_point and 'reason' in memory_point:
                        # Convert memory point to natural dialogue language（复原为原始拼接）
                        text = f"I like {memory_point['specific_item']}, {memory_point['reason']}"
                        memory_points_text.append(text)
                    else:
                        print(f"[DEBUG] Skipping invalid memory point data: {memory_point}")
            else:
                print(f"[DEBUG] memory_points is not a list: {memory_points}")
                memory_points = []  # Set as empty list as default value
            
            # Merge all memory point text
            memory_points_entries = " ".join(memory_points_text)
            
            # Prepare template variables
            template_vars = {
                'preference_type': preference_type,
                'memory_points_entries': memory_points_entries,
                'overall_description': preference_info.get('overall_description', '')
            }
            
            # Use template for formatting
            template = Template(self.templates['preferences'])
            conversation_text = template.safe_substitute(template_vars)
            
            print(f"[DEBUG] preference_type: {preference_type}")
            print(f"[DEBUG] conversation_text: {conversation_text}")
            
            # Parse to messages format
            messages = self._parse_conversation_to_messages(conversation_text)
            
            # Ensure each preference dialogue has even number of messages, if not add assistant reply
            if len(messages) % 2 != 0:
                # Add assistant reply to make even
                assistant_replies = [
                    "Good, I understand.",
                    "I see, I'll remember that.",
                    "Thank you for sharing, I remember that.",
                    "Good, I understand.",
                    "I see, this information is very helpful."
                ]
                import random
                assistant_reply = random.choice(assistant_replies)
                messages.append({
                    "role": "assistant",
                    "content": assistant_reply
                })
                print(f"[DEBUG] Added assistant reply for {preference_type} to make even: {assistant_reply}")
            
            # Add current preference dialogue as an independent list to results
            # Final postprocess to ensure cleanliness for callers like stage5
            messages = self.postprocess_message_list(messages)
            all_conversations.append(messages)
        
        return all_conversations
    
    def convert_single_persona_to_conversations(self, persona_data: Dict[str, Any], uuid: str = None) -> Dict[str, Any]:
        """
        Convert single persona data to dialogue format
        
        Args:
            persona_data: Single persona data
            uuid: Optional UUID
            
        Returns:
            Dictionary containing dialogue, using standard LLM messages format
        """
        conversations = {
            'uuid': uuid,
            'fixed_conversation': [],
            'dynamic_conversation': [],
            'preferences_conversations': []
        }
        
        # Convert fixed section (only process non-empty data)
        if persona_data.get('fixed'):
            conversations['fixed_conversation'] = self._format_fixed_conversation(persona_data['fixed'])
        else:
            conversations['fixed_conversation'] = []
        
        # Convert dynamic section (only process non-empty data)
        if persona_data.get('dynamic'):
            conversations['dynamic_conversation'] = self._format_dynamic_conversation(persona_data['dynamic'])
        else:
            conversations['dynamic_conversation'] = []
        
        # Convert preferences section (only process non-empty data)
        if persona_data.get('preferences'):
            preferences_conversations = self._format_preferences_conversation(persona_data['preferences'])
            conversations['preferences_conversations'] = preferences_conversations
        else:
            conversations['preferences_conversations'] = []
        
        # Post-process conversations text before returning
        conversations = self.postprocess_conversations(conversations)
        return conversations
    
    def convert_single_persona_and_generate_memories(self, persona_data: Dict[str, Any], uuid: str = None) -> Dict[str, Any]:
        """
        Convert single persona data to dialogue format, simultaneously generate memory points
        
        Args:
            persona_data: Single persona data
            uuid: Optional UUID
            
        Returns:
            Dictionary containing dialogue and memory points
        """
        print(f"[DEBUG] ===============================Convert single persona data to dialogue format, simultaneously generate memory points===============================")
        # Convert dialogue
        conversations = self.convert_single_persona_to_conversations(persona_data, uuid)
        
        # Generate memory points
        all_memories = self.generate_all_init_memories(persona_data)
        
        # Merge results
        result = {
            'uuid': conversations['uuid'],
            'conversations': conversations,
            'memories': all_memories
        }

        # Global postprocess to ensure downstream (e.g., stage5) gets cleaned text without extra integration
        result = self.postprocess_stage_payload(result)
        return result
    
    def save_conversations_and_memories(self, result: Dict[str, Any], output_file: str):
        """
        Save dialogue and memory points to file
        
        Args:
            result: Data containing dialogue and memory points
            output_file: Output file path
        """
        output_dir = os.path.dirname(output_file)
        print(f"[DEBUG] Saving file to directory: {os.path.abspath(output_dir)}")
        print(f"[DEBUG] Saving file: {os.path.basename(output_file)}")
        
        # Ensure payload is cleaned before persisting
        cleaned = self.postprocess_stage_payload(result)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        
        print(f"[DEBUG] File saved successfully: {output_file}")
    
    def print_conversations(self, conversations: Dict[str, Any]):
        """
        Print dialogue content
        
        Args:
            conversations: Dialogue data
        """
        print(f"UUID: {conversations['uuid']}")
        
        print("\n" + "="*50)
        print("Fixed Information Dialogue:")
        print("="*50)
        for msg in conversations['fixed_conversation']:
            print(f"{msg['role']}: {msg['content']}")
        
        print("\n" + "="*50)
        print("Dynamic Information Dialogue:")
        print("="*50)
        for msg in conversations['dynamic_conversation']:
            print(f"{msg['role']}: {msg['content']}")
        
        print("\n" + "="*50)
        print("Preference Information Dialogue:")
        print("="*50)
        for i, conv in enumerate(conversations['preferences_conversations'], 1):
            print(f"\nPreference Dialogue {i}:")
            for msg in conv:
                print(f"{msg['role']}: {msg['content']}")
    
    def generate_init_event_memories(self, persona_data: Dict, event_type: str) -> Dict:
        """
        Generate memory points for init events, without calling LLM, directly extract from profile information
        
        Args:
            persona_data: Character data
            event_type: Event type ("initial_fixed", "initial_dynamic", "initial_preference")
            
        Returns:
            Dictionary containing memory points
        """
        print(f"[DEBUG] Generating memory points for {event_type} event")
        
        # Get character name
        persona_name = "Unknown Person"
        if persona_data.get("fixed", {}).get("basic_info", {}).get("name"):
            persona_name = persona_data["fixed"]["basic_info"]["name"]
        
        # Extract corresponding profile data based on event type
        if event_type == "initial_fixed":
            profile_section = self._normalize_data_structure(persona_data.get("fixed", {}))
            memory_points = []
            
            # Basic information memory points
            basic_info = profile_section.get("basic_info", {})
            if basic_info.get('name'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"User's name is {basic_info['name']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if basic_info.get('gender'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s gender is {basic_info['gender']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if basic_info.get('birth_date'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s birth date is {basic_info['birth_date']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if basic_info.get('location'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name} lives in {basic_info['location']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            
            # Age information
            age = profile_section.get("age", {})
            if age.get('current_age'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s age in January 2025 is {age['current_age']} years old",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            
            # Education information
            education = profile_section.get("education", {})
            if education.get('highest_degree'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s highest education level is {education['highest_degree']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if education.get('major'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s major is {education['major']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            
            # Personality information
            personality = profile_section.get("personality", {})
            if personality.get('mbti'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s MBTI personality type is {personality['mbti']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            tags = personality.get("tags", [])
            if tags:
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s personality tags include: {', '.join(tags)}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            
            # Family life information (now in fixed)
            family = profile_section.get("family_life", {})
            if family.get('parent_status'):
                parent_status_map = {
                    'both_alive': 'Both parents are alive',
                    'father_alive': 'Father is alive',
                    'mother_alive': 'Mother is alive',
                    'both_deceased': 'Both parents are deceased',
                    'one_deceased': 'One parent is deceased'
                }
                status = parent_status_map.get(family['parent_status'], family['parent_status'])
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s parent status: {status}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if family.get('partner_status'):
                partner_status_map = {
                    'single': 'Single',
                    'married': 'Married',
                    'divorced': 'Divorced',
                    'widowed': 'Widowed',
                    'dating': 'Dating'
                }
                status = partner_status_map.get(family['partner_status'], family['partner_status'])
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s partner status: {status}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if family.get('child_status'):
                child_status_map = {
                    'no_children': 'No children',
                    'one_child': 'One child',
                    'two_children': 'Two children',
                    'three_children': 'Three children'
                }
                status = child_status_map.get(family['child_status'], family['child_status'])
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s child status: {status}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            
            # Life goals
            life_goal = profile_section.get("life_goal", {})
            if life_goal.get('statement'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s life goal statement: {life_goal['statement']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if life_goal.get('motivation'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s motivation for pursuing goals: {life_goal['motivation']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
            if life_goal.get('target_metrics'):
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name}'s target metrics: {life_goal['target_metrics']}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
        
        elif event_type == "initial_dynamic":
            profile_section = self._normalize_data_structure(persona_data.get("dynamic", {}))
            memory_points = []
            
            # Process merged dynamic data structure
            # Traverse all dynamic types
            for dynamic_type, dynamic_data in profile_section.items():
                if dynamic_type == "career_status":
                    # Career status memory points
                    if dynamic_data.get('employment_status'):
                        employment_status_map = {
                            'employed': 'Employed',
                            'unemployed': 'Unemployed',
                            'student': 'Student',
                            'retired': 'Retired'
                        }
                        status = employment_status_map.get(dynamic_data['employment_status'], dynamic_data['employment_status'])
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s employment status is {status}",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('industry'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name} works in the {dynamic_data['industry']} industry",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('company_name'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name} works at {dynamic_data['company_name']}",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('job_title'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s job title is {dynamic_data['job_title']}",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('monthly_income'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s monthly income is {dynamic_data['monthly_income']} USD",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('savings_amount'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s savings amount is {dynamic_data['savings_amount']} USD",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                
                elif dynamic_type == "health_status":
                    # Health status memory points
                    if dynamic_data.get('physical_health'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s physical health condition: {dynamic_data['physical_health']}",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                    if dynamic_data.get('mental_health'):
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s mental health condition: {dynamic_data['mental_health']}",
                            "memory_type": "Persona Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
                
                elif dynamic_type == "social_relationships":
                    # Social relationship memory points
                    for name, info in dynamic_data.items():
                        memory_points.append({
                            "index": len(memory_points) + 1,
                            "memory_content": f"{persona_name}'s {info['relationship_type']} {name}, {info['description']}",
                            "memory_type": "Relationship Memory",
                            "is_update": "False",
                            "original_memories": []
                        })
        
        elif event_type == "initial_preference":
            profile_section = self._normalize_data_structure(persona_data.get("preferences", {}))
            memory_points = []
            
            # Process merged preferences data structure
            # Traverse all preference types
            for preference_type, preference_info in profile_section.items():
                # Add preference type memory point
                memory_points.append({
                    "index": len(memory_points) + 1,
                    "memory_content": f"{persona_name} has preferences in {preference_type}",
                    "memory_type": "Persona Memory",
                    "is_update": "False",
                    "original_memories": []
                })
                
                # Add specific preference memory points
                for memory_point in preference_info.get('memory_points', []):
                    # Split preference and reason into two memory points
                    # First memory point: preference itself
                    memory_points.append({
                        "index": len(memory_points) + 1,
                        "memory_content": f"{persona_name} {memory_point['type_description']}: {memory_point['specific_item']}",
                        "memory_type": "Persona Memory",
                        "is_update": "False",
                        "original_memories": []
                    })
                    
                    # Second memory point: preference reason
                    memory_points.append({
                        "index": len(memory_points) + 1,
                        "memory_content": f"{persona_name} {memory_point['type_description']} {memory_point['specific_item']} reason: {memory_point['reason']}",
                        "memory_type": "Persona Memory",
                        "is_update": "False",
                        "original_memories": []
                    })
        
        # Generate dialogue information - use actual event times instead of defaults
        # These should be provided by the calling function
        start_time = None  # Should be provided by caller
        end_time = None    # Should be provided by caller
        
        # Generate dialogue summary based on event type
        if event_type == "initial_fixed":
            dialogue_summary = "User introduced their basic information to AI, including name, gender, birth date, residence, educational background, personality traits, family life, and life goals and other fixed information."
            dialogue_goal = "Understand user's basic personal information and background"
        elif event_type == "initial_dynamic":
            dialogue_summary = "User introduced their current status to AI, including career status, health status, and social relationships and other dynamic information."
            dialogue_goal = "Understand user's current life status and dynamic information"
        elif event_type == "initial_preference":
            dialogue_summary = "User introduced their various preferences to AI, including lifestyle habits, interests and hobbies, values and other personal preferences."
            dialogue_goal = "Understand user's personal preferences and habits"
        
        return {
            "start_time_point": start_time,
            "end_time_point": end_time,
            "dialogue_goal": dialogue_goal,
            "dialogue_summary": dialogue_summary,
            "memory_points": memory_points,
            "memory_points_count": len(memory_points),
            "user_motivation_type": ""
        }

    def generate_all_init_memories(self, persona_data: Dict) -> Dict[str, Dict]:
        """
        Generate memory points for all init events
        
        Args:
            persona_data: Character data
            
        Returns:
            Dictionary containing memory points for all init events
        """
        print("[DEBUG] Starting to generate memory points for all init events...")
        
        all_memories = {}
        
        # Generate memory points for three types of init events
        init_event_types = ["initial_fixed", "initial_dynamic", "initial_preference"]
        
        for event_type in init_event_types:
            print(f"[DEBUG] Generating memory points for {event_type} event...")
            memories = self.generate_init_event_memories(persona_data, event_type)
            all_memories[event_type] = memories
        
        print(f"[DEBUG] All init event memory points generation completed, generated {len(all_memories)} event types")
        return all_memories

    def print_conversations_and_memories(self, result: Dict[str, Any]):
        """
        Print dialogue and memory point content
        
        Args:
            result: Data containing dialogue and memory points
        """
        # Print dialogue
        self.print_conversations(result['conversations'])
        
        # Print memory points
        print("\n" + "="*50)
        print("Memory Point Information:")
        print("="*50)
        for event_type, memories in result['memories'].items():
            print(f"\n{event_type} Event Memory Points:")
            self.print_memories_summary(memories)

    def print_memories_summary(self, memories_data: Dict):
        """
        Print memory point summary
        
        Args:
            memories_data: Memory point data
        """
        print(f"\nMemory Point Summary:")
        print(f"Dialogue Goal: {memories_data['dialogue_goal']}")
        print(f"Dialogue Summary: {memories_data['dialogue_summary']}")
        print(f"Memory Point Count: {memories_data['memory_points_count']}")
        print(f"Time Range: {memories_data['start_time_point']} - {memories_data['end_time_point']}")
        
        print(f"\nMemory Point List:")
        for memory in memories_data['memory_points']:
            print(f"  {memory['index']}. [{memory['memory_type']}] {memory['memory_content']}")


def process_single_persona_file(input_file: str, output_file: str = None, converter: ConversationConverter = None) -> Dict[str, Any]:
    """
    Process single persona file (JSONL format, usually only one record)
    
    Args:
        input_file: Input file path
        output_file: Output file path, if None then don't save
        converter: ConversationConverter instance, if None then create new one
        
    Returns:
        Processing result dictionary
    """
    if converter is None:
        converter = ConversationConverter()
    
    print(f"[DEBUG] Starting to process file: {input_file}")
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            break  # Only process first record
    
    # Process single record
    result = converter.convert_single_persona_and_generate_memories(data['profile'], data['uuid'])
    
    # Save result
    if output_file:
        converter.save_conversations_and_memories(result, output_file)
    
    return result


def process_multiple_persona_files(input_files: List[str], output_dir: str = "output", converter: ConversationConverter = None) -> List[Dict[str, Any]]:
    """
    Batch process multiple persona files
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        converter: ConversationConverter instance, if None then create new one
        
    Returns:
        List of processing results
    """
    if converter is None:
        converter = ConversationConverter()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for input_file in input_files:
        print(f"\n[DEBUG] Processing file: {input_file}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_conversations_and_memories.json")
        
        try:
            # Process single file
            result = process_single_persona_file(input_file, output_file, converter)
            results.append(result)
            print(f"[DEBUG] File processing completed: {output_file}")
            
        except Exception as e:
            print(f"[ERROR] Error processing file {input_file}: {e}")
            continue
    
    return results


def process_jsonl_file_with_multiple_records(input_file: str, output_dir: str = "output", converter: ConversationConverter = None) -> List[Dict[str, Any]]:
    """
    Process JSONL file containing multiple records
    
    Args:
        input_file: Input file path
        output_dir: Output directory
        converter: ConversationConverter instance, if None then create new one
        
    Returns:
        List of processing results
    """
    if converter is None:
        converter = ConversationConverter()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    print(f"[DEBUG] Starting to process multi-record file: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}_record_{line_num}_conversations_and_memories.json")
                
                print(f"[DEBUG] Processing record {line_num}, UUID: {data.get('uuid', 'unknown')}")
                
                # Process single record
                result = converter.convert_single_persona_and_generate_memories(data['profile'], data['uuid'])
                
                # Save result
                converter.save_conversations_and_memories(result, output_file)
                results.append(result)
                
                print(f"[DEBUG] Record {line_num} processing completed: {output_file}")
                
            except Exception as e:
                print(f"[ERROR] Error processing record {line_num}: {e}")
                continue
    
    return results


def main():
    """Main function, demonstrate how to use converter"""
    # Create converter
    converter = ConversationConverter()
    
    # Convert data
    input_file = "data/stage1_3_preferences.jsonl"
    output_file = "data/init_event_conversations_and_memories.json"
    
    try:
        print("="*60)
        print("Convert Dialogue and Generate Memory Points")
        print("="*60)
        
        # Use new processing function
        result = process_single_persona_file(input_file, output_file, converter)
        
        # Print results
        converter.print_conversations_and_memories(result)
        
        print(f"\nProcessing completed! File saved to: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"Error occurred during conversion: {e}")


if __name__ == "__main__":
    main()
