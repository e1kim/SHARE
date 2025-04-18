import re
import json
import argparse
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from generation import model_response, response_session_prompt
from memory_selection import selection_session_prompt, ms_system
from utils.model_utils import get_peft_llama, get_peft_gemma
from utils.dialogue_utils import write_output, load_data, make_dataclass
from extraction import process_extraction_prompt, extract_batch, process_extraction
from update_module import accumulate_session_data, process_update_prompt, update_system_batch, process_update_model


'''
main.py continuously generates the entire session by creating the first two utterances, and then continuously updates!
'''
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

update_path = "chano12/update_llama"
extract_path = "eunwoneunwon/EPISODE-extraction_llama3"


def create_collate_fn(tokenizer, prompt):
    def collate_fn(batch):

        prompts = [item[prompt] for item in batch]

        tokenized_inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

        return {
            'inputs': tokenized_inputs,
            'original_batch': batch  
        }
    return collate_fn
    

def episode(data, backbone, update_method, outputfilename, total_session_number, start_num, end_num, batch_size=2):
    
    # check backbone model
    if not (backbone == 'llama' or backbone == 'gemma'):
        assert False, "No backbone model"
    
    episode_dataset = make_dataclass(data, total_session_number, start_num, end_num)
    print(f"Current data len:",len(episode_dataset))
    print(f"Backbone: {backbone} and update method: {update_method}")
    print(f"output file name: {outputfilename}")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for session_number in range(2, total_session_number):
        ############### memory update ###############     
        if update_method == "accumulate":
            for key, value in episode_dataset.items():
                previous_data = value[session_number - 2]["info"]
                current_data = value[session_number-1]["info"]
                value[session_number-1]["info"] = accumulate_session_data(previous_data, current_data)
        else:
            update_model, update_tokenizer = get_peft_llama(update_path, device)
            collate_fn = create_collate_fn(update_tokenizer, 'update_prompt')

            update_dataset = []

            # Loop over episode_dataset to collect all samples
            for key, value in episode_dataset.items():
                previous_data = value[session_number - 2]["info"]
                current_data = value[session_number-1]["info"]

                update_prompt = process_update_prompt(previous_data, current_data)
                update_dataset.append({
                    'update_prompt': update_prompt,
                    'episode_key': key,
                    'previous_data': previous_data,  # Include if needed later
                    'current_data': current_data     # Include if needed later
                })
            
            data_loader = DataLoader(update_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            for batch_idx, batch_data in enumerate(data_loader):
                print(f'Batch {batch_idx+1}/{len(data_loader)}')
                inputs = batch_data['inputs'].to(device)  # 데이터를 device로 이동
                original_batch = batch_data['original_batch']

                update_response = update_system_batch(inputs, update_model, update_tokenizer)

                for j, resp in enumerate(update_response):
                    print(original_batch)
                    episode_key = original_batch[j]['episode_key']
                    merge_session = process_update_model(original_batch[j]['previous_data'], resp)
                    print(f"LABEL for Example {batch_idx*batch_size + j}: {merge_session}")
                    episode_dataset[episode_key][session_number-1]['info'] = merge_session  # or appropriate processing


            del update_model, update_tokenizer
            torch.cuda.empty_cache()

        
        ############### memory selection & response generation ###############

        if update_method == "independent":
            memory_path = "chano12/llama_memory_selection_inden_0.5"
        else:
            memory_path = "eunwoneunwon/EPISODE-selection_llama3"
        
        ms_model, ms_tokenizer = get_peft_llama(memory_path, device)
        collate_fn = create_collate_fn(ms_tokenizer, 'ms prompt')

        
        if backbone == "llama" and update_method in ["update", "accumulate"]:
            gen_path = "chano12/llama_with_memory_response"
            gen_model, gen_tokenizer = get_peft_llama(gen_path, device)

        if backbone == "llama" and update_method in ["independent"]:
            gen_path = "chano12/llama_with_memory_response_individual"
            gen_model, gen_tokenizer = get_peft_llama(gen_path, device)

        elif backbone == "llama" and update_method in ["noshare"]:
            gen_path = "chano12/llama3_noshare_tag"
            gen_model, gen_tokenizer = get_peft_llama(gen_path, device)

        elif backbone == "gemma" and update_method in ["update", "accumulate"]:
            gen_path = "chano12/gemma_with_memory_response"
            gen_model, gen_tokenizer = get_peft_gemma(gen_path, device)
        
        elif backbone == "gemma" and update_method in ["independent"]: 
            gen_path = "chano12/gemma_with_memory_response_individual"
            gen_model, gen_tokenizer = get_peft_gemma(gen_path, device)

        elif backbone == "gemma" and update_method in ["noshare"]:
            gen_path = "chano12/gemma_noshare_tag"
            gen_model, gen_tokenizer = get_peft_gemma(gen_path, device)
        else:
            assert False, "check your model and method"

        for key, value in episode_dataset.items():
            session = value[session_number]
            previous_info = value[session_number-1]['info']
            s1_name = session['info'].s1_name
            s2_name = session['info'].s2_name
            
            dialogues = session['dialogues'][:2]
            dia_text = ""
            dia_no_tag_text = ""
            dia_text_s1 = ""
            dia_text_s2 = ""

            for dial in dialogues:
                processed_list = [s.rstrip(".") for s in dial["label"]]
                response_string = ", ".join(processed_list)
                dia_text += f"{dial['speaker']}: ({response_string}) {dial['text']}\n"
                dia_no_tag_text += f"{dial['speaker']}: {dial['text']}\n"
                if dial['speaker'] == s1_name:
                    dia_text_s1 += f"{dial['speaker']} : ({response_string}) {dial['text']}\n"
                    dia_text_s2 += f"{dial['speaker']} : {dial['text']}\n"
                else:
                    dia_text_s1 += f"{dial['speaker']} : {dial['text']}\n"
                    dia_text_s2 += f"{dial['speaker']} : ({response_string}) {dial['text']}\n"
                

            for turn in range(8):  # generate 8 utterances
                # memory selection & response generation
                next_speaker = dialogues[-2]['speaker']
                if update_method == "noshare":
                    ms_prompt = selection_session_prompt(previous_info, dia_no_tag_text, next_speaker, "noshare")
                elif update_method == "independent":
                    ms_prompt = selection_session_prompt(previous_info, dia_no_tag_text, next_speaker, "independent")
                else:
                    ms_prompt = selection_session_prompt(previous_info, dia_no_tag_text, next_speaker, "share")
                
                selected_memory_list, output = ms_system(ms_prompt, ms_model, ms_tokenizer, device)
                selected_memory = [s.rstrip(".") for s in selected_memory_list]
                selected_memory = ", ".join(selected_memory)
                if selected_memory == "" or "Everyday Language" in selected_memory:
                    selected_memory = "Everyday Language"

                # generate response
               
                if update_method == "independent":
                    if next_speaker == s1_name:
                        generate_prompt = response_session_prompt(dia_text_s1, selected_memory, next_speaker)
                    else:
                        generate_prompt = response_session_prompt(dia_text_s2, selected_memory, next_speaker) 
                else:
                    generate_prompt = response_session_prompt(dia_text, selected_memory, next_speaker)
                
                print(f"Speaker: {next_speaker}\nGenerate Prompt:", generate_prompt)
                response = model_response(generate_prompt, gen_model, gen_tokenizer, device)
                print("-"*100)
                print(f"Response: {response}")
                print("-"*100)
                # append generated response to dialogues
                dialogues.append({
                    "speaker": next_speaker,
                    "text": response,
                    "label": selected_memory_list
                })
                dia_no_tag_text += f"{next_speaker}: {response}\n"
                dia_text += f"{next_speaker}: ({selected_memory}) {response}\n"
                if next_speaker == s1_name:
                    dia_text_s1 += f"{next_speaker}: ({selected_memory}) {response}\n"
                    dia_text_s2 += f"{next_speaker}: {response}\n"
                else:
                    dia_text_s1 += f"{next_speaker}: {response}\n"
                    dia_text_s2 += f"{next_speaker}: ({selected_memory}) {response}\n"
                
            session['dialogues'] = dialogues
            value[session_number]['dia_text'] = dia_text
            value[session_number]['dia_no_tag_text'] = dia_no_tag_text
            print(dia_text)

        del gen_model, gen_tokenizer, ms_model, ms_tokenizer
        torch.cuda.empty_cache()

        ############### information extraction ###############
        ext_model, ext_tokenizer = get_peft_llama(extract_path, device)

        collate_fn = create_collate_fn(ext_tokenizer, 'ie prompt')
        ie_dataset = []

        for key, value in episode_dataset.items():
            dia_no_tag_text = value[session_number]['dia_no_tag_text']
            ie_prompt = process_extraction_prompt(value[session_number]["info"], dia_no_tag_text)
            ie_dataset.append({
                'ie_prompt': ie_prompt,
                'episode_key': key
            })
        collate_fn = create_collate_fn(ext_tokenizer, 'ie_prompt')
        data_loader = DataLoader(ie_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        for batch_idx, batch_data in enumerate(data_loader):
            print(f'Batch {batch_idx+1}/{len(data_loader)}')
            inputs = batch_data['inputs'].to(device)  # 데이터를 device로 이동
            original_batch = batch_data['original_batch']
            information = extract_batch(inputs, ext_model, ext_tokenizer, device)

            for j, info in enumerate(information):
                episode_key = original_batch[j]['episode_key']
                final_info = process_extraction(episode_dataset[episode_key][session_number]['info'], info)
                print(f"LABEL for Example {batch_idx*batch_size + j}: {final_info}")
                print("-"*100)
                episode_dataset[episode_key][session_number]['info'] = final_info


        del ext_model, ext_tokenizer
        torch.cuda.empty_cache()

        # save dataset by session
        save_dataset = []
        for key, value in episode_dataset.items():
            info = asdict(value[session_number]["info"])
            save_data = {
                "number": session_number+1,
                "dia_text": value[session_number]['dia_text'],
                "dia_no_tag_text": value[session_number]['dia_no_tag_text'],
                "info": info,
                "previous info": asdict(value[session_number-1]["info"])
            }
            save_dataset.append({key: save_data})

        write_output(save_dataset, outputfilename + f"_{session_number+1}_session.json")


def main():
    parser = argparse.ArgumentParser(description="json to excel file")
    parser.add_argument("--method", type=str, help="The method")
    parser.add_argument("--backbone", type=str, help="The backbone model")
    parser.add_argument("--input_file", type=str, help="The input file")
    parser.add_argument("--output_file", type=str, help="The output file")
    parser.add_argument("--startnumber", type=int, help="The start session number")
    parser.add_argument("--endnumber", type=int, help="The end session number")
    parser.add_argument(
        "--total_session_number", type=int, help="The total session number"
    )
    args = parser.parse_args()
    filtered_data = load_data(args.input_file, args.total_session_number)

    # start mode #

    if args.method == "accumulate":
        episode(
            filtered_data,
            args.backbone,
            args.method,
            args.output_file,
            args.total_session_number,
            args.startnumber,
            args.endnumber
        )
    elif args.method in ["update", "noshare", "independent"]:
        episode(
            filtered_data,
            args.backbone,
            args.method,
            args.output_file,
            args.total_session_number,
            args.startnumber,
            args.endnumber
        )
    else:
        assert False, "No method error!!"

if __name__ == "__main__":
    main()
