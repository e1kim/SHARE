from utils.model_utils import other_generate_batch, other_generate
import re
from dataclasses import dataclass

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


def ext_prompt(speaker1, speaker2, dialogues_text):
    prompt = f"""You are a conversation analyst tasked with examining two conversations.

In your analysis, categorize the dialogue based on five criteria:
1. **Persona Information**: Discuss aspects such as personality, job, age, education, favorite foods, music, hobbies, family life, daily activities, health, etc.
2. **Temporary event**: Identify information that will soon become irrelevant, such as upcoming deadlines like "I need to submit my assignment by Friday" or temporary states like "I have a cold."
3. **Shared Memory**: Focus on past experiences that the speakers refer to during their conversation, which they have previously experienced together. This category includes both explicitly mentioned memories and those implied through their dialogue. For example, the exchange 'Alice: Wasn’t that jazz festival we went to last summer amazing?' 'Bob: It was phenomenal, especially the live band under the stars.' should be categorized here because it indicates that Alice and Bob shared the experience of attending a jazz festival together.
4. **Mutual Event**: This category captures significant events and interactions occurring directly between {speaker1} and {speaker2} during the current conversation, excluding any third-party involvement. Consider only those interactions that are substantial and directly involve both speakers. For example, from the exchange "Alice: Aren't these shoes pretty?", "Bob: Try them on.", "Alice: How do they look? Do they suit me?", you can extract that "Alice and Bob are experiencing shopping together."
5. **None**: Assign this category to parts of the conversation that do not fit into the above categories.

Proceed to analyze the dialogue, addressing it one turn at a time:
{dialogues_text}

Your task is to extract:
- Persona information for {speaker1}
- Persona information for {speaker2}
- Temporary event for {speaker1}
- Temporary event for {speaker2}
- Shared memories between {speaker1} and {speaker2}
- Mutual events occurring during the conversation between {speaker1} and {speaker2}

Format your findings by separating each category with '***'. If no information is found for a category, indicate it with 'None'. The expected format is:
[***Persona: {speaker1}'s information or 'None'***Persona: {speaker2}'s information or 'None'***Temporary: {speaker1}'s event or 'None'***Temporary: {speaker2}'s event or 'None'***Shared Memory: information or 'None'***Mutual Event: information or 'None'***]
Present your responses directly, using the speakers' names without pronouns and avoiding category labels. For instance, rather than stating "***{speaker1}'s temporary event includes an upcoming math project due tomorrow.***", simply note "***Temporary: {speaker1} has a math project due tomorrow.***"
Ensure that each analysis output is succinct, covering only the essential elements of the dialogue. Ensure you cover every part of the dialogue comprehensively. 
If a specific category does not apply, move on to the next without mention. Your detailed analysis will help illuminate the nuances of their interactions, capturing the essence of their shared and immediate experiences within the current dialogue.
Answer:"""

    return prompt


def extract_data(text):

    pattern = re.compile(r"Answer:\s*(.*)")
    match = pattern.search(text)

    if match:
        answer = match.group(1).strip()
    else:
        print("No answer found")
        return None
    return answer


def extract_batch(inputs, model, tokenizer, device):
    output = other_generate_batch(
        model,
        tokenizer,
        inputs,
        num_beams=1,
        num_beam_groups=1,
        do_sample=True,
        num_return_sequences=1,
        max_new_tokens=3000,
    )
    response = [extract_data(out) for out in output]

    return response



def return_sentences(text):
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Jr|Sr|Prof|MR|MRS|MS|DR|JR|SR|PROF|COL|SGT|LT|CPL)\.",
        r"\1<dot>",
        text,
    )
    text = re.sub(r"\b([A-Za-z])\.([A-Za-z])\.", r"\1<dot>\2<dot>", text)

    sentence_endings = re.compile(r"(?<=\.|\?|!)\s")
    sentences = sentence_endings.split(text.strip())

    sentences = [sentence.replace("<dot>", ".") for sentence in sentences]
    exclusion_phrases = [
        "There is no",
        "There are no",
        "information is not",
        "information cannot be",
        "None",
        "No shared",
        "No temporal information",
        "no temporal information",
        "no information for",
        "no shared",
        "No specific information about",
    ]
    sentences = [
        sentence
        for sentence in sentences
        if not any(phrase in sentence for phrase in exclusion_phrases)
    ]


    return sentences


def split_information(text):

    print(f"{text}\n\n")

    patterns = {
        "p1": r"\*\*\*Persona: (.*?)\*\*\*",
        "p2": r"\*\*\*Persona:.*?\*\*\*Persona: (.*?)\*\*\*",
        "t1": r"\*\*\*Temporary: (.*?)\*\*\*",
        "t2": r"\*\*\*Temporary:.*?\*\*\*Temporary: (.*?)\*\*\*",
        "shared": r"\*\*\*Shared Memory: (.*?)\*\*\*",
        "mutual": r"\*\*\*Mutual Event: (.*?)(?:\*\*\*|\*\*\*\]|\*+)$"
    }
    

    extracted_info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted_info[key] = return_sentences(match.group(1))
        else:
            extracted_info[key] = []

    return extracted_info

def process_extraction(session_data, extracted_information):

    extracted_info = split_information(extracted_information)

    newSessionInformation = SessionInformation(
        p1=extracted_info["p1"],
        p2=extracted_info["p2"],
        t1=extracted_info["t1"],
        t2=extracted_info["t2"],
        shared=extracted_info["shared"],
        mutual=extracted_info["mutual"],
    )
    session_data.information = newSessionInformation

    return session_data


def process_extraction_prompt(session_data, dialogues):
    s1_name = session_data.s1_name
    s2_name = session_data.s2_name

    extraction_prompt = ext_prompt(s1_name, s2_name, dialogues)

    return extraction_prompt