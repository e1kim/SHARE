import os, sys
import re
import json
import torch
from dataclasses import dataclass
from utils.model_utils import update_generate_batch, update_generate

current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)

@dataclass
class SessionInformation:
    p1: list
    p2: list
    t1: list
    t2: list
    shared: list
    mutual: list

@dataclass
class SessionData:
    s1_name: str
    s2_name: str
    information: SessionInformation

def return_prompt(previous_memory, current_memory):
    prompt = f"""\
You are a language expert who understands the flow of conversation and manages memory. To effectively manage memory in a conversational system, it is crucial to understand the memory itself. As the conversation progresses, compare the information from previous sessions with the current session to update the memory and remove unnecessary sentences. Memory is categorized into the following four types:
1. Persona information: This captures essential characteristics, including personality, occupation, and interests.
2. Personal event: This information covers transient details like impending deadlines or current health conditions.
3. Mutual event: This captures significant interactions between the speakers, focusing on substantial events directly involving both individuals. Over time, these mutual events become new shared memories.
4. Shared memory: This refers to past experiences or memories that the two speakers have shared together prior to the current conversational context.
Guidelines for Memory Management:
Tasks to Perform in the Current Session:
1. Remove incomplete information: Remove sentences that are incomplete or do not clearly convey the context.
    * Example: "SAM is interested in something." or "SAM mentions a place he visited."
2. Remove information not suitable for conversation topics: Remove information that is irrelevant to the main topic of conversation.
    * Example: "JANE remembers SAM." or “JANE has a need to urinate.”
3. Remove unrelated personal events: Remove personal event information that is not directly related to the individual or does not influence the conversation flow.
    * Example: "MARK talked about a coworker who went on vacation last month."
4. Remove duplicate information: If the same information is provided in both Persona and Personal events, or if the same information is provided in Persona and Shared memory, remove the Persona and retain the other information.
    * Example: “KATE enjoys watching movies.” (Persona) and “KATE often watches movies on weekends.” (Personal event) provide similar information, so remove the Persona.
    * Example: “MIKE remembers the trip to Paris.” (Persona) and “MIKE and JANE shared a memorable trip to Paris.” (Shared memory) are similar; remove the Persona.
5. Update Persona based on Mutual events: Update the Persona with emotions or reactions caused by Mutual events, and write sentences in the past tense.
    * Example: The Persona "JACK feels betrayed and angry." should be updated to "SARAH told JACK about her secret involvement in a rival project, causing JACK to feel betrayed and angry."
Methods for Memory Update:
1. Connect sequential/causal events: Link and update events that are sequential or have a causal relationship.
    * Example:
        * Previous memory:
            * Tom recently got a new job.
            * Tom was very nervous on his first day at work.
        * Current memory:
            * Tom successfully completed his first project at the new job.
        * Updated memory:
            * Tom recently got a new job and was very nervous on his first day. 
            * Tom has since successfully completed his first project.
2. Update conflicting events: Reflect changes or transitions when the previous and current memories contain conflicting information.
    * Example:
        * Previous memory:
            * Ellie did not enjoy her recent trip.
            * Ellie said she would no longer plan trips.
        * Current memory:
            * Ellie is planning a trip with her friends.
            * Ellie is looking forward to traveling again.
        * Updated memory:
            * Ellie did not enjoy her recent trip, but now she is planning a new trip with friends and is looking forward to it.
3. Remove unnecessary personal event information: Exclude any unnecessary details about personal events. If the personal event only reflects a very short-term, trivial state (such as someone being in transit), it should be removed.
• Example: "Jay is on the bus" should be removed.
4. Accumulate unrelated events: Accumulate personal events that do not fit guidelines 1 through 3.
    * Example:
        * Previous memory:
            * JANE likes spicy food.
        * Current memory:
            * JANE dislikes math.
        * Updated memory:
            * JANE likes spicy food.
            * JANE dislikes math.
5. Use the past tense for Mutual events: Mutual events from the current session become past events, so convert them to the past tense.
    * Example:
        * Previous memory:
            * John and Alice are planning a trip together.
        * Current memory:
            * John and Alice have finalized the details of their trip.
        * Updated memory:
            * John and Alice planned a trip together and have finalized the details.
Actual Content Update:
Use the following structure to update the memory based on the provided guidelines.
All sentences in the updated memory must start with a person’s name.
Previous memory:
{previous_memory}
Current memory:
{current_memory}
Updated memory:"""
	
    return prompt

def return_memory(info):
    def process_list(lst):
        if not lst:  
            return ''
        return '\n'.join(['- ' + string for string in lst])

    # 정보를 처리하는 부분
    persona1 = process_list(info.information.p1)
    persona2 = process_list(info.information.p2)
    personal1 = process_list(info.information.t1)
    personal2 = process_list(info.information.t2)
    shared_memory = process_list(info.information.shared)
    mutual_memory = process_list(info.information.mutual)

    memory = f"""Persona:
{persona1}
{persona2}
Personal event:
{personal1}
{personal2}
Shared memory:
{shared_memory}
Mutual event:
{mutual_memory}"""
        
    return memory

# Function to add sentences to the appropriate dictionary
def add_to_dict(category_dict, sentence):
    # Extract the first word as the name and convert it to lowercase
    name = sentence.split()[0].upper()
    if name not in category_dict:
        category_dict[name] = []
    category_dict[name].append(sentence)

# Function to add sentences to the shared memory list
def add_to_list(memory_list, sentence):
    memory_list.append(sentence)

# Function to parse the input string
def parse_memory(update_response, persona_info, personal_event_info, shared_memory_list):
    current_category = None
    # If the string starts with 'Updated memory :', ignore that part and only process the rest
    if update_response.startswith("Updated memory :"):
        update_response = update_response[len("Updated memory :"):].strip()

    for line in update_response.splitlines():
        line = line.strip()
        if line == "":
            continue
        if line.endswith(":"):
            current_category = line[:-1].strip().lower()
        elif re.match(r"^-\s?", line):  # This regex matches lines starting with '-' followed by optional space
            sentence = line[1:].strip()  # Remove the dash and any leading/trailing spaces
            if current_category == "persona":
                add_to_dict(persona_info, sentence)
            elif current_category == "personal event":
                add_to_dict(personal_event_info, sentence)
            elif current_category == "shared memory" or current_category == "mutual event":
                add_to_list(shared_memory_list, sentence)
    
    return persona_info, personal_event_info, shared_memory_list

def replace_names_with_uppercase(response, name1, name2):
    """
    Replace occurrences of name1 and name2 in response with their uppercase versions.
    This function assumes that name1 and name2 are already in uppercase.
    """
    def replace_name(match):
        matched_name = match.group(0)
        if matched_name.lower() == name1.lower():
            return name1  # replace with uppercase version of s1_name
        elif matched_name.lower() == name2.lower():
            return name2  # replace with uppercase version of s2_name
        return matched_name

    # Use regex to find all occurrences of name1 and name2 and replace them
    pattern = re.compile(r'\b(' + re.escape(name1.lower()) + r'|' + re.escape(name2.lower()) + r')\b', re.IGNORECASE)
    updated_response = pattern.sub(replace_name, response)

    return updated_response

def extract_data(text):

    answer = text.rsplit("Updated memory:", 1)[-1].strip()

    return answer

def process_update_prompt(previous_data, current_data):
    previous_memory = return_memory(previous_data)
    current_memory = return_memory(current_data)
    update_prompt = return_prompt(previous_memory, current_memory)

    return update_prompt

def update_system_batch(inputs, model, tokenizer):
    # get gpt response
    output = update_generate_batch(
            model,
            tokenizer,
            inputs,
            num_beams=3,
            num_return_sequences=1,
            max_new_tokens=3000,
        )
    
    update_response = [extract_data(out) for out in output]

    return update_response
     
def process_update_model(previous_data, update_response):
    # Define dictionaries to store categorized information
    
    persona_info = {}
    personal_event_info = {}
    shared_memory_list = []  # List to store shared memory and mutual event information
    s1_name = previous_data.s1_name
    s2_name = previous_data.s2_name

    # Update the response to replace the names with uppercase names
    update_response = replace_names_with_uppercase(update_response, s1_name, s2_name)

    # Parse the input memory and populate the dictionaries
    persona_info, personal_event_info, shared_memory_list = parse_memory(
        update_response, persona_info, personal_event_info, shared_memory_list
    )
    # Use .get() method to safely access dictionary values or return a default value if key does not exist
    p1_info = persona_info.get(s1_name, [])
    p2_info = persona_info.get(s2_name, [])
    t1_info = personal_event_info.get(s1_name, [])
    t2_info = personal_event_info.get(s2_name, [])

    merged_info = SessionInformation(
        p1 = p1_info,
        p2 = p2_info,
        t1 = t1_info,
        t2 = t2_info,
        shared = shared_memory_list,
        mutual = []
    )

    merged_session = SessionData(
        s1_name = s1_name,
        s2_name = s2_name,
        information = merged_info
    )


    return merged_session


    
def accumulate_session_data(previous_data: SessionData, current_data: SessionData) -> SessionData:
    merged_info = SessionInformation(
        p1 = previous_data.information.p1 + current_data.information.p1,
        p2 = previous_data.information.p2 + current_data.information.p2,
        t1 = previous_data.information.t1 + current_data.information.t1,
        t2 = previous_data.information.t2 + current_data.information.t2,
        shared = previous_data.information.shared + current_data.information.shared + current_data.information.mutual,
        mutual = []
    )

    merged_session = SessionData(
        s1_name = previous_data.s1_name,
        s2_name = previous_data.s2_name,
        information = merged_info
    )

    return merged_session