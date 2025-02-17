from dataclasses import dataclass, asdict
import torch
from generation import return_model_no_tag_prompt, model_response
from utils.model_utils import get_base_llama, get_base_gemma
from utils.dialogue_utils import write_output, make_dataclass, make_dialogues

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



def process_response_generation(session, gen_model, gen_tokenizer, device):
    dia_text, dia_no_tag_text = make_dialogues(session)

    last = session["dialogues"][-1]
    last_speaker = last["speaker"]
    generate_prompt = return_model_no_tag_prompt(dia_no_tag_text)

    last_utter = f"{last_speaker}: "
    generate_prompt = generate_prompt.rstrip() + "\n" + last_utter

    dialogue_response = model_response(
        generate_prompt, gen_model, gen_tokenizer, device
    )
    dia_no_tag_text += f"{last_speaker}: {dialogue_response}\n"

    return dialogue_response, dia_no_tag_text, generate_prompt


def baseline_episode(data, backbone, outputfilename, total_session_number):
    episode_dataset = make_dataclass(data, total_session_number)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ############### response generation ###############
    if backbone == "llama":
        gen_path = "meta-llama/Llama-3.1-8B-Instruct"
        gen_model, gen_tokenizer = get_base_llama(gen_path, device)
    elif backbone == "gemma":
        gen_path = "google/gemma-2b"
        gen_model, gen_tokenizer = get_base_gemma(gen_path, device)
    else:
        print("No backbone model")
        assert False
    
    for i in range(1, total_session_number):
        
        for key, value in episode_dataset.items():
            generate_response, dialogues, prompt = process_response_generation(
                value[i], gen_model, gen_tokenizer, device
            )
            value[i]["model response"] = generate_response
            value[i]["real answer"] = value[i]["dialogues"][-1]["text"]
            value[i]["generated dialogues"] = dialogues
            value[i]['generate prompt'] = prompt
            print("-" * 100)
            print(f"model response: {generate_response}")

        # save dataset by session
        save_dataset = []
        for key, value in episode_dataset.items():
            info = asdict(value[i]["info"])
            save_data = {
                "number": i,
                "real answer": value[i]["real answer"],
                "model response": value[i]["model response"],
                "generate prompt": value[i]['generate prompt'],
                "info": info,
            }
            save_dataset.append({key: save_data})

        write_output(save_dataset, outputfilename + f"_{i}_session.json")


def main():
    pass

if __name__ == "__main__":
    main()
