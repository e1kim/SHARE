from peft import (
    PeftModel,
    PeftConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
    encoder_repetition_penalty=1,
    repetition_penalty=1,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        attention_mask=inputs.attention_mask,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        encoder_repetition_penalty=encoder_repetition_penalty,
        repetition_penalty=repetition_penalty,
        min_length=15,
        no_repeat_ngram_size=3,
        pad_token_id = tokenizer.pad_token_id
    )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result

def generate_batch(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
    encoder_repetition_penalty=1,
    repetition_penalty=1,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        attention_mask=inputs.attention_mask,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        encoder_repetition_penalty=encoder_repetition_penalty,
        repetition_penalty=repetition_penalty,
        min_length=15,
        no_repeat_ngram_size=3,
        pad_token_id = tokenizer.pad_token_id
    )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return result

def ms_generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):


    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,  
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=tokenizer.pad_token_id) 

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]

    return result


def ms_generate_batch(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):


    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,  
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=tokenizer.pad_token_id) 

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)

    return result

def other_generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result

def other_generate_batch(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )

    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return result

def update_generate(
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
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id = tokenizer.pad_token_id
        )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return result

def update_generate_batch(
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
        attention_mask=inputs.attention_mask,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        pad_token_id = tokenizer.pad_token_id
        )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return result

def get_peft_llama(path, device):

    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)

    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,load_in_8bit=True)
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()
    
    return model, tokenizer


def get_peft_gemma(path, device):
    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, fast_tokenizer=True
    )
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()

    return model, tokenizer


def get_base_llama(path, device):

    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    model.eval()

    return model, tokenizer

def get_base_gemma(path, device):
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    model.eval()

    return model, tokenizer