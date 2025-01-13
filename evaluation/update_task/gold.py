from dataclasses import dataclass, asdict
import torch
from generation import model_response
from utils.model_utils import get_peft_llama, get_peft_gemma
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

def return_model_prompt(dia_text):
    prompt = f"""
Task: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:

1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.
2. Everyday language dialogues: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.

**Dialogue History**:
{dia_text}
"""
    return prompt
def process_response_generation(session, gen_model, gen_tokenizer, device, memory):
    dia_text, dia_no_tag_text = make_dialogues(session)

    last = session["dialogues"][-1]
    last_speaker = last["speaker"]
    generate_prompt = return_model_prompt(dia_text)

    last_utter = f"{last_speaker}: ({memory})"
    generate_prompt = generate_prompt.rstrip() + "\n" + last_utter

    dialogue_response = model_response(
        generate_prompt, gen_model, gen_tokenizer, device
    )
    dia_no_tag_text += f"{last_speaker}: {dialogue_response}\n"

    return dialogue_response, dia_no_tag_text, generate_prompt


def gold_episode(data, backbone, outputfilename, total_session_number):
	episode_dataset = make_dataclass(data, total_session_number)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	############### response generation ###############
	if backbone == "llama":
		gen_path = "chano12/llama3_with_4000tag_v5"
		gen_model, gen_tokenizer = get_peft_llama(gen_path, device)
	elif backbone == "gemma":
		gen_path = "chano12/gemma_with_4000tag_v5"
		gen_model, gen_tokenizer = get_peft_gemma(gen_path, device)
	else:
		print("No backbone model")
		assert False
	
	for i in range(1, total_session_number):
		for key, value in episode_dataset.items():
			value[i]["real select"] = value[i]['dialogues'][-1]["label"]
			generate_response, dialogues, prompt = process_response_generation(
				value[i], gen_model, gen_tokenizer, device, value[i]["real select"]
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
				"real select": value[i]["real select"],
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
