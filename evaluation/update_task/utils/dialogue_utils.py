import json
from dataclasses import dataclass
import os

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

def write_output(dataset, outputfilename):
    output_dir = os.path.dirname(outputfilename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(outputfilename, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def make_dialogues(session):
    dia_text = ""
    dia_no_tag_text = ""
    dialogue_set = session["dialogues"][:-1]

    for dial in dialogue_set:
        processed_list = [s.rstrip(".") for s in dial["label"]]
        response_string = ", ".join(processed_list)

        dia_text += f"{dial['speaker']}: ({response_string}) {dial['text']}\n"
        dia_no_tag_text += f"{dial['speaker']}: {dial['text']}\n"

    return dia_text, dia_no_tag_text


def load_data(path, total_session_number):
    with open(path, "r") as file:
        data = json.load(file)

    filtered_data = {}
    for key, value in data.items():
        if "dialogue" in value and len(value["dialogue"]) >= total_session_number:
            filtered_data[key] = value

    return filtered_data


def parse_speakers(key):
    return tuple(key.replace("(", "").replace(")", "").replace("'", "").split(", "))


def make_dataclass(data, total_session_number, start=0, end=200):
    episode_dataset = {}
    count = 0

    # setting data
    for key, value in data.items():
        if count < start:
            count += 1
            continue
        if count >= end:
            break

        conversations = value["dialogue"]
        s1_name, s2_name = parse_speakers(key)
    
        dialogue = []

        first_session = conversations[0]
        Information = SessionInformation(
            p1=first_session[f"{s1_name}'s persona"],
            p2=first_session[f"{s2_name}'s persona"],
            t1=first_session[f"{s1_name}'s temporary event"],
            t2=first_session[f"{s2_name}'s temporary event"],
            shared=first_session["Shared memory"],
            mutual=first_session["Mutual event"],
        )
        session_Data = SessionData(
            s1_name=s1_name,
            s2_name=s2_name,
            information=Information,
        )
        session_data = {
            "number": 1,
            "dialogues": first_session["dialogues"],
            "info": (session_Data),
        }
        dialogue.append(session_data)

        seconde_session = conversations[1]
        Information = SessionInformation(
            p1=seconde_session[f"{s1_name}'s persona"],
            p2=seconde_session[f"{s2_name}'s persona"],
            t1=seconde_session[f"{s1_name}'s temporary event"],
            t2=seconde_session[f"{s2_name}'s temporary event"],
            shared=seconde_session["Shared memory"],
            mutual=seconde_session["Mutual event"],
        )
        session_Data = SessionData(
            s1_name=s1_name,
            s2_name=s2_name,
            information=Information,
        )
        session_data2 = {
            "number": 2,
            "dialogues": seconde_session["dialogues"],
            "info": (session_Data),
        }
        dialogue.append(session_data2)

        for idx, session in enumerate(conversations[2:total_session_number]):
            Information = SessionInformation(
                p1=[], p2=[], t1=[], t2=[], shared=[], mutual=[]
            )
            session_Data = SessionData(
                s1_name=s1_name,
                s2_name=s2_name,
                information=Information,
            )
            session_data = {
                "number": idx + 3,
                "dialogues": session["dialogues"],
                "info": (session_Data),
            }
            dialogue.append(session_data)

        episode_dataset[key] = dialogue
        count += 1

    print("total data num:", count)

    return episode_dataset