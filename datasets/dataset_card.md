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

