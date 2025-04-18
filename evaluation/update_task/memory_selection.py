from utils.model_utils import ms_generate_batch, ms_generate
import re
from dataclasses import dataclass
from utils.dialogue_utils import make_dialogues
import random

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


def process_memory_selection(speaker1, speaker2, session: SessionInformation):
    info_list = [session.p1, session.p2, session.t2, session.t1, session.shared]
    candidates = ["Everyday Language"] + [
        item for info in info_list if info is not None for item in info
    ]
    random.shuffle(candidates)
    candidates_string = "\n".join(f"- {value}" for idx, value in enumerate(candidates))

    return candidates_string, candidates

def return_ms_prompt(dialogues_text, candidates, next_speaker):
    prompt = f"""\
You are a conversation analyst. \
You need to understand the context well and predict the next part of the dialogue.
Based on the provided candidate memories and dialogue history, select all the appropriate memories for the next part of the conversation. \
These memories are elements that form the basis of the conversation.
If no suitable memories are available, choose 'Everyday Language,' which refers to common, everyday expressions.

Task:
Candidate Memories:
{candidates}

Dialogue History:
{dialogues_text} 
Select all the appropriate memories for the next part of the conversation by {next_speaker}. \
If there are two or more memories, separate them with '###':"""

    return prompt

def extract_data(text):
    # Adjusted regular expression
    pattern = re.compile(r"separate them with '###':\s*(.*)", re.IGNORECASE)
    match = pattern.search(text)

    if match:
        return match.group(1).strip()
    return None


def ms_system_batch(inputs, model, tokenizer, device):
    output = ms_generate_batch(model, tokenizer, inputs, num_beams=1, num_return_sequences=1, max_new_tokens=400)
    
    memory = [extract_data(out) for out in output]

    memory_list = [mem.split('###') for mem in memory]

    return memory_list, memory


def process_selection_prompt(
    data: SessionData,
    session
):
    dia_text, dia_no_tag_text = make_dialogues(session)
    next_speaker = session["dialogues"][-1]['speaker']
    
    information = data.information
    s1_name = data.s1_name
    s2_name = data.s2_name

    candidates_string, candidates = process_memory_selection(
        s1_name, s2_name, information
    )
    ms_prompt = return_ms_prompt(dia_no_tag_text, candidates_string, next_speaker)

    return ms_prompt

def selection_session_prompt(
    data: SessionData,
    dia_no_tag_text,
    next_speaker,
    method,
):
    
    information = data.information
    s1_name = data.s1_name
    s2_name = data.s2_name

    if method == "noshare":
        info_list = [information.p1, information.t1, information.p2, information.t2]
        candidates = ["Everyday Language"] + [
            item for info in info_list if info is not None for item in info
        ]
        random.shuffle(candidates)
        candidates_string = "\n".join(f"- {value}" for idx, value in enumerate(candidates))
        
    elif method == "share":
        info_list = [information.p1, information.t1, information.p2, information.t2, information.shared]
        candidates = ["Everyday Language"] + [
            item for info in info_list if info is not None for item in info
        ]
        random.shuffle(candidates)
        candidates_string = "\n".join(f"- {value}" for idx, value in enumerate(candidates))
    
    elif method == "independent":
        if next_speaker == s1_name:
            info_list = [information.p1, information.t1, information.shared]
        else:
            info_list = [information.p2, information.t2, information.shared]
        candidates = ["Everyday Language"] + [
            item for info in info_list if info is not None for item in info
        ]
        random.shuffle(candidates)
        candidates_string = "\n".join(f"- {value}" for idx, value in enumerate(candidates))

    else:
        assert False, "share or noshare?"
    
    ms_prompt = return_ms_prompt(dia_no_tag_text, candidates_string, next_speaker)

    return ms_prompt


def ms_system(prompt, model, tokenizer, device):
    input__ = prompt
    input_ = tokenizer(input__, return_tensors="pt").to(device)
    output = ms_generate(
        model,
        tokenizer,
        input_,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=500,
    )

    memory = extract_data(output)
    memory_list = memory.split("###")

    return memory_list, output