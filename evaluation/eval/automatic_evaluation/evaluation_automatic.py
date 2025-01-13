import os, sys, json
from datetime import datetime
import pandas as pd

current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)
from utils.eval_utils import get_ppl, read_json_file, get_result
from utils.model_utils import (
    get_base_model,
    get_peft_gemma,
    get_peft_llama,
    get_base_llama)

import torch
import argparse
from evaluate import load

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=3,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        encoder_repetition_penalty=0.8,
        repetition_penalty = 1.5,
        min_length = 15
        )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument(
        "--model_name", type=str, help="Choose the Model which you want to use"
    )
    parser.add_argument("--task_name", type=str, help="task_name")
    parser.add_argument("--data_path", type=str, help="data_path")
    parser.add_argument("--model_tag", type=str, help='tag')
    args = parser.parse_args()
    return args

def extract_data_without_tag(prompt):
    prompts = prompt["prompt"].rstrip() + "\n"

    last_utter = f"{prompt['last_speaker']}:"

    dialogue = prompts + last_utter
    ppl_input = prompts + last_utter + f" {prompt['answer']}"

    return ppl_input, dialogue, prompt["last_speaker"], prompt["answer"]


def extract_data_with_tag(prompt,tag):
    prompts = prompt["prompt"].rstrip() + "\n"
    last_utter = f"{prompt['last_speaker']}: ({prompt[tag]})"

    dialogue = prompts + last_utter

    ppl_input = prompts + last_utter + f" {prompt['answer']}"

    return ppl_input, dialogue, prompt["last_speaker"], prompt[tag], prompt["answer"]



def get_model(task_name, model_name):        
    print(task_name, model_name)
    if task_name == "wo_tag":
        path = "_without_tag"
    elif task_name == "w_tag":
        path = "_with_tag"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == "llama":
        print(model_name)
        path = f"chano12/{model_name}3" + path
        model, tokenizer = get_peft_llama(path, device)
        model.config.pad_token_id = model.config.eos_token_id[0]
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]

        model.eval()

    elif model_name == "base_llama":
        path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(model_name)
        model, tokenizer = get_base_llama(path, device)
        model.eval()

    elif model_name == "gemma":
        path = f"chano12/{model_name}" + path
        print(model_name)
        model, tokenizer = get_peft_gemma(path, device)
        model.eval()

    elif model_name == "base_gemma":
        path = "google/gemma-2b"
        print(model_name)
        model, tokenizer = get_base_model(path, device)
        model.eval()

    else:
        print("Problem to load the model.")
    
    return model, tokenizer


def evaluation_chat_system(
    name, prompt, model, tokenizer, device, bert_eval, rouge_eval, tag = None
):

    if name == "baseline":
        print("This model is baseline llama")
        print()
        print(prompt["prompt"])
        (
            ppl_input,
            input__,
            person,
            utterance,
        ) = extract_data_without_tag(prompt)

        print(f'Last word -> {person} : "{utterance}"')

        input_ = tokenizer(input__, return_tensors="pt").to(device)
        output = generate(
            model,
            tokenizer,
            input_,
            num_beams=3,
            num_return_sequences=1,
            max_new_tokens=100,
        )

        ### utterance : correct answer
        ### response : Model-generated answer

        response = output.replace(input__, "")
        response = response.split("\n")[0]

        print(f"prediction : {response}")
        print(f"Real answer : {utterance}")

        reference = [utterance.split()]
        candidate = response.split()

        output_list = [response.strip()]
        last_utter_list = [utterance.strip()]
        # evalation
        bert_score = bert_eval.compute(
            predictions=output_list, references=last_utter_list, lang="en"
        )
        rouge_score = rouge_eval.compute(
            predictions=output_list, references=last_utter_list
        )

        ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )

        ### ppl
        ppl = get_ppl(ppl_input, model, tokenizer, device)

        print(f"Bert Score : {bert_score}")
        print(f"Rouge Score : {rouge_score}")
        print(f"bleu 1/2 : {bleu_unigram} {bleu_bigram}")
        print(f"bleu 3/4 : {bleu_trigram} {bleu_fourgram}")
        print(f"ppl : {ppl}")


        return bert_score, rouge_score, bleu_unigram, bleu_bigram, bleu_trigram, bleu_fourgram, response, ppl

    if name == "wo_tag":
        print("This is a wo tag evaluation")
        print()
        print(prompt["prompt"])
        (
            ppl_input,
            input__,
            person,
            utterance,
        ) = extract_data_without_tag(prompt)
        print(f'Last word -> {person} : "{utterance}"')

        input_ = tokenizer(input__, return_tensors="pt").to(device)
        output = generate(
            model,
            tokenizer,
            input_,
            num_beams=3,
            num_return_sequences=1,
            max_new_tokens=100,
        )

        ### utterance : correct answer
        ### response : Model-generated answer

        response = output.replace(input__, "")
        response = response.split("\n")[0]
        print(f"prediction : {response}")
        print(f"Real answer : {utterance}")

        reference = [utterance.split()]
        candidate = response.split()

        output_list = [response.strip()]
        last_utter_list = [utterance.strip()]
        # evalation
        bert_score = bert_eval.compute(
            predictions=output_list, references=last_utter_list, lang="en"
        )
        rouge_score = rouge_eval.compute(
            predictions=output_list, references=last_utter_list
        )

        ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )


        ### ppl
        ppl = get_ppl(ppl_input, model, tokenizer, device)

        print(f"Bert Score : {bert_score}")
        print(f"Rouge Score : {rouge_score}")
        print(f"bleu 1/2 : {bleu_unigram} {bleu_bigram}")
        print(f"bleu 3/4 : {bleu_trigram} {bleu_fourgram}")
        print(f"ppl : {ppl}")

        return bert_score, rouge_score, bleu_unigram, bleu_bigram, bleu_trigram, bleu_fourgram, response, ppl

    if name == "w_tag":
        print("This is with tag evaluation")
        print()
        ppl_input, input__, person, trait, utterance = extract_data_with_tag(
            prompt, tag
        )
        print(f'Last word -> {person} : ({trait}) "{utterance}"')

        input_ = tokenizer(input__, return_tensors="pt").to(device)
        output = generate(
            model,
            tokenizer,
            input_,
            num_beams=3,
            num_return_sequences=1,
            max_new_tokens=100,
        )

        ### utterance : correct answer
        ### response : Model-generated answer

        response = output.replace(input__, "").split("\n")[0]

        print(f"prediction : {response}")
        print(f"Real answer : {utterance}")

        output_list = [response.strip()]
        last_utter_list = [utterance.strip()]

        reference = [utterance.split()]
        candidate = response.split()

        # evalation
        bert_score = bert_eval.compute(
            predictions=output_list, references=last_utter_list, lang="en"
        )
        rouge_score = rouge_eval.compute(
            predictions=output_list, references=last_utter_list
        )

        ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )

        ### ppl
        ppl = get_ppl(ppl_input, model, tokenizer, device)

        print(f"Bert Score : {bert_score}")
        print(f"Rouge Score : {rouge_score}")
        print(f"bleu 1/2 : {bleu_unigram} {bleu_bigram}")
        print(f"bleu 3/4 : {bleu_trigram} {bleu_fourgram}")
        print(f"ppl : {ppl}")

        return bert_score, rouge_score, bleu_unigram, bleu_bigram, bleu_trigram, bleu_fourgram, response, ppl


def main():
    args = parse_args()
    print(args.task_name, args.model_name)

    model, tokenizer = get_model(args.task_name, args.model_name)

    print("evaluation start")

    data = read_json_file(args.data_path)
    print(f"# of data : {len(data)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    bert_list = []
    rouge_list = []
    bleu_1_list = []
    bleu_2_list = []
    bleu_3_list = []
    bleu_4_list = []
    infer_list = []
    ppl_list = []

    bertscore_eval = load("bertscore")
    rouge_eval = load("rouge")

    try:
        for num, prompt in enumerate(data):


            print(num)
            (
                bert_score,
                rouge_score,
                bleu_1_score,
                bleu_2_score,
                bleu_3_score,
                bleu_4_score,
                infer_sentence,
                ppl,
            ) = evaluation_chat_system(
                args.task_name,
                prompt,
                model,
                tokenizer,
                device,
                bertscore_eval,
                rouge_eval,
                args.model_tag
            )
            bert_list.append(bert_score)
            rouge_list.append(rouge_score)
            bleu_1_list.append(bleu_1_score)
            bleu_2_list.append(bleu_2_score)
            bleu_3_list.append(bleu_3_score)
            bleu_4_list.append(bleu_4_score)
            infer_list.append(infer_sentence)
            ppl_list.append(ppl)
            prompt['generate_sentence'] = infer_sentence
 
            result = {
                'responses': infer_list,
                'bert_scores': [{'precision': b} for b in bert_list],
                'rouge_scores': rouge_list,
                'bleu_scores': {
                    'unigram': bleu_1_list,
                    'bigram': bleu_2_list,
                    'trigram': bleu_3_list,
                    'fourgram': bleu_4_list,
                },
                'ppl_values': ppl_list,
            }

        
            # print(num)
            # (
            #     bert_score,
            #     rouge1,
            #     rouge2,
            #     rougeL,
            #     rougeLsum,
            #     bleu_1_score,
            #     bleu_2_score,
            #     bleu_3_score,
            #     bleu_4_score,
            #     distinct_1,
            #     distinct_2,
            #     ppl,
            # ) = get_result(
            #     bert_list, rouge_list, bleu_1_list, bleu_2_list,bleu_3_list,bleu_4_list, infer_list, ppl_list
            # )

            # print(
            #     f"PPL : {ppl}, \nBertScore : {bert_score} \nrouge1 : {rouge1} \nrouge2 : {rouge2} \nrougeL : {rougeL} \nrougeLsum : {rougeLsum}, \nbleu_1 : {bleu_1_score} \nbleu_2 : {bleu_2_score} "
            # )
            # print(f'bleu_3 : {bleu_3_score} \nbleu_4 : {bleu_4_score}')
            # print(f"distinct 1/2 {distinct_1, distinct_2}")
    except:
            (
            bert_score,
            rouge1,
            rouge2,
            rougeL,
            rougeLsum,
            bleu_1_score,
            bleu_2_score,
            bleu_3_score,
            bleu_4_score,
            distinct_1,
            distinct_2,
            ppl,
        ) = get_result(
            result
        )

            print(
            f"PPL : {ppl}, \nBertScore : {bert_score} \nrouge1 : {rouge1} \nrouge2 : {rouge2} \nrougeL : {rougeL} \nrougeLsum : {rougeLsum}, \nbleu_1 : {bleu_1_score} \nbleu_2 : {bleu_2_score} "
        )
            print(f'bleu_3 : {bleu_3_score} \nbleu_4 : {bleu_4_score}')
            print(f"distinct 1/2 {distinct_1, distinct_2}")
            finish = num

    print(result)

    (   bert_score,
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
        bleu_1_score,
        bleu_2_score,
        bleu_3_score,
        bleu_4_score,
        distinct_1,
        distinct_2,
        ppl,
    ) = get_result(
        result
    )


    print(
        f"PPL : {ppl}, \nBertScore : {bert_score} \nrouge1 : {rouge1} \nrouge2 : {rouge2} \nrougeL : {rougeL} \nrougeLsum : {rougeLsum}, \nbleu_1 : {bleu_1_score} \nbleu_2 : {bleu_2_score} "
    )
    print(f'bleu_3 : {bleu_3_score} \nbleu_4 : {bleu_4_score}')
    print(f"distinct 1/2 {distinct_1, distinct_2}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dict = {
        'name': f'{args.task_name, args.data_path, args.model_name, args.model_tag, num}',
        'ppl': round(ppl, 4),
        'bleu-1/2': f'{round(bleu_1_score, 4)}/{round(bleu_2_score, 4)}',
        'bleu-3/4': f'{round(bleu_3_score, 4)}/{round(bleu_4_score, 4)}',
        'rouge-1/2': f'{round(rouge1, 4)}/{round(rouge2, 4)}',
        'rougeL': round(rougeL, 4),
        'bert_score': round(bert_score, 4),
        'distinct_1/2': f'{round(distinct_1, 4)}/{round(distinct_2, 4)}'
    }
    

    filename = f"per_session_v6_{current_time}.json"
    
    with open(filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)

if __name__ == "__main__":
    main()
