# SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script

**SHARE** is a novel long-term dialogue dataset constructed from movie scripts. 

## Dataset Overview

This dataset consists of:
- **Conversations**: Dialogue exchanges between two main characters in various movie scripts.
- **Annotations**: Detailed extractions using GPT-4, including:
  - **Persona**: Persona information captures essential characteristics, including personality, occupation, and interest.
  - **Temporary Events**: Personal event information covers transient details like impending deadlines or current health conditions.
  - **Shared Memory**: Shared memory refers to past memories that the two individuals have shared together prior to the current conversational context.
  - **Mutual Memory**: Mutual event information captures significant interactions between the speakers, focusing on substantial events directly involving both individuals. Over time, these mutual events become new shared memories.

## Purpose

SHARE is designed to:
1. Enhance the study of **long-term dialogues** by leveraging shared memories between participants.
2. Serve as a benchmark for developing dialogue models capable of managing and utilizing shared memories effectively.

## Dataset Structure
The dataset is organized as a JSON file, structured as follows:

### Top-level Keys
Each key in the dataset represents a tuple of two characters (`('Character1', 'Character2')`) from a specific movie. The corresponding value contains the metadata and conversation details between the two characters.

### JSON Structure

```json
{
    "('Character1', 'Character2')": {
        "movie": "Movie Title",
        "dialogue": [
            {
                "session": int,  # Dialogue session number
                "dialogues": [
                    {
                        "speaker": "Speaker Name",  # Name of the speaker
                        "text": "Dialogue text",  # Utterance spoken by the speaker
                        "label": [  # Semantic labels for the utterance
                            "Label describing the utterance context or speaker behavior."
                        ],
                        "utterance": int  # Order of the utterance in the session
                    }
                ],
                "Character1's persona": [  # Persona traits inferred for Character1
                    "Trait or characteristic of Character1."
                ],
                "Character2's persona": [  # Persona traits inferred for Character2
                    "Trait or characteristic of Character2."
                ],
                "Character1's temporary event": [  # Temporary context/event for Character1
                    "Context or event affecting Character1 during the session."
                ],
                "Character2's temporary event": [  # Temporary context/event for Character2
                    "Context or event affecting Character2 during the session."
                ],
                "Shared memory": [  # Explicitly shared memories between the characters
                    "Memory explicitly shared between Character1 and Character2."
                ],
                "Mutual event": [  # Mutually experienced events or inferred shared context
                    "Event or context that both characters participate in."
                ]
            }
        ]
    }
}

## Citation

If you use this dataset, please cite the following paper:

> [SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script](https://arxiv.org/pdf/2410.20682)

@article{kim2024share,
  title={SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script},
  author={Kim, Eunwon and Park, Chanho and Chang, Buru},
  journal={arXiv preprint arXiv:2410.20682},
  year={2024}
}