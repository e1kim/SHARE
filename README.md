### This is the official project of paper: [SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script](https://arxiv.org/pdf/2410.20682)

# Overview
![Figure Description](figure/Framework.png)

**SHARE** is a novel long-term dialogue dataset constructed from movie scripts, designed to enhance conversations by leveraging shared memories between individuals. It includes persona information, event summaries, and both explicit and implicit shared memories to enrich dialogue engagement. Additionally, we propose **EPISODE**, a dialogue framework that utilizes these shared experiences to make long-term conversations more engaging and sustainable.


## Datasets
<img src="figure/Dataset_statistics.png" alt="Figure Description" width="500">

### 📦 Download the Dataset

You can download this dataset directly from Hugging Face:  
👉 [https://huggingface.co/datasets/eunwoneunwon/SHARE](https://huggingface.co/datasets/eunwoneunwon/SHARE)

### 🔽 How to Download

```python
mkdir -p datasets/
cd datasets/
git lfs install
git clone https://huggingface.co/datasets/eunwoneunwon/SHARE
```

You can explore the SHARE dataset, which is organized into `train`, `validation`, and `test` splits under the `data/` directory.

Each split is stored as a separate JSON file:
```
data/ 
 └── train.json 
 └── valid.json
 └── test.json
```
### 💬 Example Dialogue

Below is a sample from the `valid.json` split of the SHARE dataset:

```json
{
  "session": 3,
  "dialogues": [
    {
      "speaker": "BERADA",
      "text": "I got you another six months. I told them it takes time.",
      "label": [
        "BERADA has ensured an extension of six months for the operation."
      ]
    },
    {
      "speaker": "DONNIE",
      "text": "Same budget?",
      "label": [
        "DONNIE and BERADA share past interactions concerning the operation, which involves managing a budget for an ongoing operation."
      ]
    },
    {
      "speaker": "BERADA",
      "text": "Same budget. Look, Joe, not that I don't see any movement, but--do you see any movement? I got my neck out on this.",
      "label": [
        "BERADA is responsible for managing the operation and feels pressure due to a lack of visible progress."
      ]
    },
    {
      "speaker": "DONNIE",
      "text": "Whatever it takes, I'm gonna get these bastards.",
      "label": [
        "DONNIE is dedicated to his mission and willing to do whatever it takes."
      ]
    }
  ]
}

```


## Model Download

Below are the Hugging Face model links used in this project. You can easily access and download them by clicking on the model names.

### 1. **Selection Model**
- [Selection Model](https://huggingface.co/eunwoneunwon/EPISODE-selection_llama3)

### 2. **Generation Models**
- [Generation Llama Model](https://huggingface.co/chano12/llama3_with_tag)
- [Generation Gemma Model](https://huggingface.co/chano12/gemma_with_tag)

### 3. **Extraction Model**
- [Extraction Model](https://huggingface.co/eunwoneunwon/EPISODE-extraction_llama3)

### 4. **Update Model**
- [Update Model](https://huggingface.co/chano12/update_llama)


## Training Code Reference

The implementation of the generation, selection, extraction, and update models in this project is inspired by the following repository:

- [Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations](https://github.com/nuochenpku/COMEDY.git)

Special thanks to the authors for providing an excellent foundation for our work.


# Installation
Clone this repository and install the required packages:
```bash
conda env create -f environment.yml
conda activate share
```
## Evaluation

### Automatic Evaluation
This section describes the code for creating an evaluation dataset.

1. Run `automatic_eval.sh` inside the `update_task` folder.
2. The entire system's modules will execute, generating the final response.

For more details about automatic evaluation, refer to the `eval/automatic_evaluation` folder.

---

### Multi-Session Evaluation
This section explains the code for multi-session evaluation.

1. Run `multi_session_eval.sh` inside the `update_task` folder.
2. Then, execute `multi_eval.sh` in the `evaluation/eval` folder to perform GPT evaluation.

**Note:** When running GPT evaluation, make sure to set the `OPENAI_API_KEY` environment variable.

---

### EPISODE Evaluation
This section provides instructions for EPISODE evaluation.

1. Run `multi_session_eval.sh` inside the `update_task` folder.
2. Then, execute `episode_share.sh` in the `evaluation/eval` folder to perform GPT evaluation.
**Note:** When running GPT evaluation, make sure to set the `OPENAI_API_KEY` environment variable.

# Citation

If you use this project in your research, please cite it as follows:

