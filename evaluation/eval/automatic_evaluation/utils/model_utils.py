# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from transformers.deepspeed import HfDeepSpeedConfig

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

def create_model(model_name_or_path):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model_config.dropout = 0.0

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=model_config, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    return model, tokenizer

def get_lora_model(model_name, model_path_, ds_config):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    if model_name == "llama":
        print(f"This model is {model_name}")

        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        model, tokenizer = create_model(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_name == "quan_gemma":
        print(f"This model is {model_name}")
        model_path = "google/gemma-7b"

        tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_name == "gemma":

        def create_model(model_name_or_path):
            model_config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            model_config.dropout = 0.0

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, config=model_config, trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

            return model, tokenizer

        print(f"This model is {model_name}")
        model_path = "google/gemma-2b"

        model, tokenizer = create_model(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_name == "quan_llama":
        print(f"This model is {model_name}")
        model_path = "beomi/Llama-3-Open-Ko-8B"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )        

        tokenizer = AutoTokenizer.from_pretrained(model_path, fast_tokenizer=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 , quantization_config=bnb_config
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        print("bring a checkpoint")
        config = PeftConfig.from_pretrained(model_path_)

        def create_model(model_name_or_path):
            model_config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            model_config.dropout = 0.0
            
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,torch_dtype=torch.bfloat16 ,quantization_config=bnb_config
            )

            model = prepare_model_for_kbit_training(model)


            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

            return model, tokenizer

        model, tokenizer = create_model(config.base_model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(model, model_path_, is_trainable=True)
        model.print_trainable_parameters()

    return model, tokenizer


### Infernece model.


def get_peft_llama(path, device):
    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, fast_tokenizer=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, load_in_8bit=True
    )
    model = PeftModel.from_pretrained(model, path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    tokenizer.padding_side = 'left'

    model.to(device)

    return model, tokenizer


def get_peft_gemma(path, device):

    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, fast_tokenizer=True
    )
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, path)
    tokenizer.padding_side = 'left'

    model.to(device)

    return model, tokenizer


def get_base_llama(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True)
    tokenizer.padding_side = 'left'
    #model.to(device)

    return model, tokenizer


def get_base_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer.padding_side = 'left'

    model.to(device)

    return model, tokenizer


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
        min_length = 15,
        pad_token_id = tokenizer.pad_token_id
        )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result