# Evaluation Script Guide

This document provides a guide for configuring and running the `evaluation_automatic.py` script. Detailed explanations of each configuration parameter and instructions are provided below.

## Environment Variable Setup

```bash
export CUDA_VISIBLE_DEVICES=2 # Specify the GPU device to use (e.g., 0, 1)
```

### Key Configuration Parameters

Below are the key parameters required to run the script and their descriptions.

1. TASK_NAME
   - Specifies the type of task to evaluate.
   - Available options:
     - wo_tag: Task without using tags
     - w_tag: Task using tags
     - baseline: Baseline task
   - Choose the appropriate value based on the purpose of your experiment.

2. MODEL_NAME
   - Specifies the model name to evaluate.
   - Available options:
     - llama: LLAMA model
     - gemma: GEMMA model
     - base_llama: Base LLAMA model
     - base_gemma: Base GEMMA model
   - Set the model name you intend to use for evaluation.

3. DATA_PATH
   - Specifies the path to the test dataset for evaluation.
   - Example: ../../../datasets/SHARE.json
   - Ensure the path points to a valid JSON file containing the test dataset.

4. MODEL_TAG
   - Specifies the type of tags to use for evaluation.
   - Available options:
     - gold_tag: Use the last utterance's tag already present in the dialogue
     - model_tag: Use the tag predicted by the selection model
   - Set the appropriate value based on the experiment requirements.

Here is an example of configuration settings for a specific experiment:

- GPU Device: GPU 2
- Task: Task using tags (w_tag)
- Model: LLAMA (llama)
- Dataset Path: ../../../datasets/SHARE.json
- Tag Type: gold_tag

Example Command:
```bash
export CUDA_VISIBLE_DEVICES=2
TASK_NAME='w_tag'
MODEL_NAME='llama'
DATA_PATH='../../../datasets/SHARE.json'
MODEL_TAG='gold_tag'

python evaluation_automatic.py --task_name $TASK_NAME --model_name $MODEL_NAME --data_path $DATA_PATH --model_tag $MODEL_TAG
```

### Model Explanation

llama3_with_tag: A model generating responses using both context and tags.
llama3_without_tag: A model generating responses using only context, without tags.
gemma_with_tag: A model generating responses utilizing context and tags.
gemma_without_tag: A model generating responses based solely on context, without tags.
update_llama: A llama model used specifically for updates.
llama3_noshare_tag: A model trained without shared memory in the learning process.
gemma_noshare_tag: A model trained without shared memory in its architecture.