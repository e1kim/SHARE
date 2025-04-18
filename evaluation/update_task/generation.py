from utils.model_utils import generate, generate_batch
from utils.dialogue_utils import make_dialogues
import unicodedata
import torch

def return_model_prompt(dia_text):
    prompt = f"""
Task: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:

1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.
2. Everyday language dialogues: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.

**Dialogue History**:
{dia_text}
"""
    return prompt


def response_generation_prompt(memory,dialogue):
    return f"Your task is to generate the next response in a dialogue by focusing on the given memory. Carefully read, understand, and retain the given memory to logically generate the next response that aligns with the provided context. Responses should be tailored according to the type of memories provided:\n1. Memory-driven dialogues: If the memory details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context. When generating a response, the provided memory must be seamlessly integrated and directly reflected in the very next reply. The given memories should be linked as related concepts and expressed naturally within a single sentence.\n2. Everyday language dialogues: If the memory is labeled \"Everyday Language,\" generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.\nMultiple memories may be provided. Thoroughly analyze all given memories and generate a response that integrates them coherently within the provided context\nGiven Memories:\n{memory}\nDialogue History:\n{dialogue}.\nresponse :\n"


def return_model_no_tag_prompt(dia_text):
    # no memory in Dialogue History

    prompt = f"""
Task: Generate the next response in the dialogue based on the provided history. The response should logically follow and predict the next reply considering the context of the conversation.

**Dialogue History**:
{dia_text}
"""
    
    return prompt

def gen_extract_data(prompt, output):

    response = output.replace(prompt, "").split("\n")[0].strip()

    response = ''.join(c for c in response if unicodedata.category(c) != 'So')

    response = response.encode('utf-8', errors='replace').decode('utf-8')
    return response

def response_batch(inputs, model, tokenizer, device):

    outputs = generate_batch(
        model,
        tokenizer,
        inputs,
        num_beams=3,
        num_return_sequences=1,
        max_new_tokens=200,
        encoder_repetition_penalty=0.8,
        repetition_penalty=1.4,
    )

    return outputs

def process_response_prompt(session, memory):
    dia_text, dia_no_tag_text = make_dialogues(session)

    last = session["dialogues"][-1]
    last_speaker = last["speaker"]
    generate_prompt = return_model_prompt(dia_text)

    last_utter = f"{last_speaker}: ({memory})"
    generate_prompt = generate_prompt.rstrip() + "\n" + last_utter

    return generate_prompt

def response_session_prompt(dia_text, selected_memory, next_speaker):

    generate_prompt = return_model_prompt(dia_text)

    last_utter = f"{next_speaker}: ({selected_memory})"
    generate_prompt = generate_prompt.rstrip() + "\n" + last_utter

    return generate_prompt


def response_notag_prompt(dia_text, next_speaker):

    generate_prompt = return_model_no_tag_prompt(dia_text)

    last_utter = f"{next_speaker}:"
    generate_prompt = generate_prompt.rstrip() + "\n" + last_utter

    return generate_prompt


def model_response(prompt, model, tokenizer, device):

    input_ = tokenizer(prompt, return_tensors="pt").to(device)
    output = generate(
        model,
        tokenizer,
        input_,
        num_beams=3,
        num_return_sequences=1,
        encoder_repetition_penalty=0.8,
        repetition_penalty = 2.0,
        max_new_tokens=200,
    )

    response = output.replace(prompt, "").split("\n")[0].strip()

    # Post-processing step to remove invalid characters
    response = ''.join(c for c in response if unicodedata.category(c) != 'So')

    # Encode to UTF-8 and decode to prevent encoding issues
    response = response.encode('utf-8', errors='replace').decode('utf-8')

    return response

