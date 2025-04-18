import re
import json
import argparse
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from generation import process_response_prompt, response_batch, gen_extract_data
from memory_selection import process_selection_prompt, ms_system_batch
from utils.model_utils import get_peft_llama, get_peft_gemma
from utils.dialogue_utils import make_dialogues, write_output, load_data, make_dataclass
from extraction import process_extraction_prompt, extract_batch, process_extraction
from update_module import accumulate_session_data, process_update_prompt, update_system_batch, process_update_model
from baseline import baseline_episode
from notag import notag_episode
from gold import gold_episode

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
memory_path = "eunwoneunwon/EPISODE-selection_llama3"
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
    

def episode(data, backbone, update_method, outputfilename, total_session_number, start_num, end_num, batch_size=4):
    
    # check backbone model
    if not (backbone == 'llama' or backbone == 'gemma'):
        assert False, "No backbone model"
    
    episode_dataset = make_dataclass(data, total_session_number, start_num, end_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(2, total_session_number):
        ############### memory update ###############     
        if update_method == "accumulate":
            for key, value in episode_dataset.items():
                previous_data = value[i - 2]["info"]
                current_data = value[i-1]["info"]
                value[i-1]["info"] = accumulate_session_data(previous_data, current_data)
        else:
            update_model, update_tokenizer = get_peft_llama(update_path, device)
            collate_fn = create_collate_fn(update_tokenizer, 'update_prompt')

            update_dataset = []

            # Loop over episode_dataset to collect all samples
            for key, value in episode_dataset.items():
                previous_data = value[i - 2]["info"]
                current_data = value[i-1]["info"]

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
                    episode_key = original_batch[j]['episode_key']
                    merge_session = process_update_model(original_batch[j]['previous_data'], resp)
                    print(f"LABEL for Example {batch_idx*batch_size + j}: {merge_session}")
                    episode_dataset[episode_key][i-1]['info'] = merge_session  # or appropriate processing


            del update_model, update_tokenizer
            torch.cuda.empty_cache()

        
        ############### memory selection ###############

        ms_model, ms_tokenizer = get_peft_llama(memory_path, device)
        collate_fn = create_collate_fn(ms_tokenizer, 'ms prompt')
        ms_dataset = []
        
        for key, value in episode_dataset.items():
            ms_prompt= process_selection_prompt(
                value[i - 1]["info"], value[i]
            )
            print(ms_prompt)
            ms_dataset.append({
                'ms prompt': ms_prompt,
                'episode_key': key  # 각 에피소드의 키 저장
            })

        data_loader = DataLoader(ms_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        for batch_idx, batch_data in enumerate(data_loader):
            print(f'Batch {batch_idx+1}/{len(data_loader)}')
            
            inputs = batch_data['inputs'].to(device)  # 데이터를 device로 이동
            original_batch = batch_data['original_batch']

            response, memory = ms_system_batch(inputs, ms_model, ms_tokenizer, device)
            
            for j, resp in enumerate(response):
                processed_list = [s.rstrip('.') for s in resp]
                response_string = ', '.join(processed_list)

                print(f"LABEL for Example {batch_idx*batch_size + j}: {response_string}")

                #save
                episode_key = original_batch[j]['episode_key']
                episode_dataset[episode_key][i]['memory select'] = response_string
                episode_dataset[episode_key][i]['ms prompt'] =  original_batch[j]['ms prompt']


        del ms_model, ms_tokenizer
        torch.cuda.empty_cache()
        

        ############### response generation ###############
        if backbone == "llama":
            gen_path = "chano12/llama3_with_tag"
            gen_model, gen_tokenizer = get_peft_llama(gen_path, device)
        else:
            gen_path = "chano12/gemma_with_tag"
            gen_model, gen_tokenizer = get_peft_gemma(gen_path, device)

        gen_dataset = []
        for key, value in episode_dataset.items():
            generate_prompt = process_response_prompt(
                value[i], value[i]['memory select']
            )
            gen_dataset.append({
                'generate prompt': generate_prompt,
                'episode_key': key
            })
        
        collate_fn = create_collate_fn(gen_tokenizer, 'generate prompt')
        data_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        for batch_idx, batch_data in enumerate(data_loader):
            print(f'Batch {batch_idx+1}/{len(data_loader)}')
            inputs = batch_data['inputs'].to(device)  # 데이터를 device로 이동
            original_batch = batch_data['original_batch']
            generate_response = response_batch(inputs, gen_model, gen_tokenizer, device)
            
            for j, resp in enumerate(generate_response):
                response = gen_extract_data(original_batch[j]['generate prompt'],resp)
                print(f"LABEL for Example {batch_idx*batch_size + j}: {response}")

                episode_key = original_batch[j]['episode_key']

                #save
                episode_dataset[episode_key][i]['model response'] = response
                episode_dataset[episode_key][i]['real answer'] = episode_dataset[episode_key][i]["dialogues"][-1]["text"]
                episode_dataset[episode_key][i]['generate prompt'] = original_batch[j]['generate prompt']

        del gen_model, gen_tokenizer
        torch.cuda.empty_cache()


        ############### information extraction ###############
        ext_model, ext_tokenizer = get_peft_llama(extract_path, device)

        collate_fn = create_collate_fn(ext_tokenizer, 'ie prompt')
        ie_dataset = []

        for key, value in episode_dataset.items():
            dia_no_tag_text = make_dialogues(value[i])
            ie_prompt = process_extraction_prompt(value[i]["info"], dia_no_tag_text)
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

                final_info = process_extraction(episode_dataset[episode_key][i]['info'], info)
                print(f"LABEL for Example {batch_idx*batch_size + j}: {final_info}")
                episode_dataset[episode_key][i]['info'] = final_info

 

        del ext_model, ext_tokenizer
        torch.cuda.empty_cache()

        # save dataset by session
        save_dataset = []
        for key, value in episode_dataset.items():
            info = asdict(value[i]["info"])
            save_data = {
                "number": i,
                "model select": value[i]["memory select"],
                "real answer": value[i]["real answer"],
                "model response": value[i]["model response"],
                "generate prompt": value[i]['generate prompt'],
                "ms prompt": value[i]["ms prompt"],
                "info": info,
                "previous info": asdict(value[i-1]["info"])
            }
            save_dataset.append({key: save_data})

        write_output(save_dataset, outputfilename + f"_{i+1}_session.json")


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

    # start mode
    if args.method == "baseline":
        baseline_episode(
            filtered_data,
            args.backbone,
            args.output_file,
            args.total_session_number,
        )
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
    elif args.method == "update":
        episode(
            filtered_data,
            args.backbone,
            args.method,
            args.output_file,
            args.total_session_number,
            args.startnumber,
            args.endnumber
        )
    elif args.method == "notag":
        notag_episode(
            filtered_data,
            args.backbone,
            args.output_file,
            args.total_session_number
        )
    elif args.method == "gold":
        gold_episode(
            filtered_data,
            args.backbone,
            args.output_file,
            args.total_session_number
        )
    else:
        assert False, "No method error!!"

if __name__ == "__main__":
    main()
