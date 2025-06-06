from openai import OpenAI
import os, sys
import json
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
import argparse
import re


def consistency(dialogue1, dialogue2, dialogue3, dialogue4, dialogue5):
    text=f"""\
Your task is to evaluate the "Consistency" between the following five consecutive conversations. Read through all the conversations carefully and assess whether the relationship between the participants remains consistent throughout the dialogue. Pay particular attention to how shifts in tone or behavior are well-supported by the context or emotional developments, allowing the relationship to evolve naturally. Even if the dialogue features notable changes in the characters' tone or attitude, these shifts should still be seen as consistent if they are explained or connected to previous conversations.
Evaluation Criteria:
Consistency measures the evolving nature of the relationship between the participants across the conversations. Conversations with good consistency will feel natural, even if the relationship experiences shifts in tone or intensity, as long as these shifts are well-supported by prior interactions or context. Shifts that show character growth or new revelations, and that maintain logical progression based on the overall relationship dynamics, should contribute positively to the consistency score.
3 Points: The conversations exhibit excellent consistency, with all shifts in tone or behavior feeling logical and well-supported by prior dialogue. In these conversations, even as the tone or intensity changes, these shifts are always justified by emotional or contextual developments between the characters. The relationship evolves naturally and feels enriched by the deeper emotional understanding of both participants. Even if the characters' actions or speech become more direct or intense, the progression makes sense within the established dynamics of their relationship.
2 Points: The conversations have moderate consistency. The relationship evolves in a mostly coherent manner, though there may be minor shifts that slightly disrupt the flow.
1 Point: The conversations show some consistency, but the relationship between the participants experiences a few unexplained or awkward shifts.
0 Points: The conversations lack consistency, with abrupt shifts in tone or behavior that feel unsupported or out of context.
Output Format:
Score : [score]
Dialogue 1 :
{dialogue1}
Dialogue 2 :
{dialogue2}
Dialogue 3 :
{dialogue3}
Dialogue 4 :
{dialogue4}
Dialogue 5 :
{dialogue5}
Score :"""
    return text

def reflectiveness(dialogue1, dialogue2, dialogue3, dialogue4, dialogue5):
    text = f"""\
Your task is to evaluate the "Reflectiveness" between the participants across the following five consecutive conversations. Reflectiveness measures how well the relationship between the participants can be inferred from their dialogue. Focus on whether the conversations naturally reveal their connection, providing clear and consistent clues about how they relate to each other.
Evaluation Criteria:
Reflectiveness evaluates the extent to which the relationship between the two participants is revealed in a way that allows for a clear understanding of their connection. A higher score is given when the conversations display detailed personal interactions, emotional exchanges, or shared experiences that suggest familiarity, trust, or an ongoing relationship. Even subtle hints, such as indirect references to shared histories or emotional support, will be considered strong indicators of Reflectiveness.
The nature of the relationship may also be inferred from consistent communication styles, recurring topics, or moments where one participant shows care, understanding, or concern toward the other, even when not explicitly stated. Reflectiveness focuses on how naturally the dialogue portrays these elements, whether through verbal cues, mutual engagement, or deeper conversations that provide insight into their bond.
Scoring Criteria:
3 points: The conversations clearly and consistently reveal the participants' relationship. Their interactions, whether direct or subtle, provide consistent cues about trust, emotional engagement, or shared history. These exchanges make it easy to define their relationship, with personal details, familiarity, or mutual understanding naturally emerging throughout the conversations.
2 points: The conversations give some insight into the relationship but lack strong, consistent cues. There is a moderate sense of connection, but not enough clarity about their bond.
1 point: The conversations provide limited or vague information, making it hard to infer the participants' relationship.
0 points: The conversations provide no clear indications of the participants' relationship, leaving their connection undefined.
Output Format:
Score : [score]
Dialogue 1 : 
{dialogue1}
Dialogue 2 : 
{dialogue2}
Dialogue 3 : 
{dialogue3}
Dialogue 4 : 
{dialogue4}
Dialogue 5 : 
{dialogue5}
Score :"""
    return text



def engagingness(dialogue1, dialogue2, dialogue3, dialogue4, dialogue5):
    text = f"""\
Your task is to evaluate the "Engagingness" between the participants in the following five consecutive conversations. Read through all the conversations carefully and assess how entertaining and engaging the dialogue is for the participants. Pay particular attention to whether the conversations are connected across sessions and if this connection enhances the overall fun and interest of the dialogue. A clear and engaging flow between sessions can increase the overall Engagingness of the conversation.
Evaluation Criteria:
Engagingness
Engagingness measures how fun and captivating the conversations are. High levels of engagingness will make the dialogue feel lively, entertaining, and enjoyable for both participants. Additionally, conversations that show a strong connection across sessions, enhancing the overall flow and enjoyment, will have higher levels of engagingness.
0 points: The conversations are not engaging. The interactions are dull, monotonous, and fail to capture any interest. There is no meaningful connection between sessions.
1 point: The conversations show minimal engagingness. There are a few moments of interest, but the overall dialogue feels flat and lacks energy. The connection between sessions is weak or nonexistent.
2 points: The conversations are moderately engaging. The participants manage to create some entertaining exchanges, but the overall flow isn't consistently fun or lively. Connections between sessions exist but are not particularly strong or noticeable.
3 points: The conversations are highly engaging, with dynamic, fun, and captivating exchanges that keep the conversation entertaining and full of energy throughout. The dialogue flows well between sessions, and this connection makes the overall experience more enjoyable.
Output Format:
Score : [score]
Dialogue 1 :
{dialogue1}
Dialogue 2 :
{dialogue2}
Dialogue 3 :
{dialogue3}
Dialogue 4 :
{dialogue4}
Dialogue 5 :
{dialogue5}
Score :"""
    return text


def load_session(filename):
	try:
		with open(filename, "r", encoding="utf-8") as f:
			last_session_data = {}
			for line in f:
				data = json.loads(line)
				last_session_data.update(data)
		return last_session_data
	except :
		print(f"다른 방법으로 시도하거나, 파일을 확인하세요.")
		return -1

def load_data(path):
	with open(path, "r") as file:
		data = json.load(file)

	return data



def get_gpt_response(prompt):
	try:
		response = client.chat.completions.create(
			model="gpt-4o-2024-08-06",
			messages=[
				{
				"role": "system",
				"content": [
					{
					"type": "text",
					"text": "You are a conversation evaluator. You need to read the dialogue carefully and evaluate it according to the evaluation criteria.\nThe output format should be as follows:\nScore: 0 or\nScore: 1 or\nScore: 2 or\nScore: 3\n\nRead and understand the evaluation criteria and the conversation carefully, then evaluate it."
					}
				]
				},
				{"role": "user", "content": f"{prompt}"},
			],
			temperature=1,
			max_tokens=1000,
			top_p=1,
			frequency_penalty=0,
			presence_penalty=0,
		)

	except Exception as e:
		error_str = str(e)
		print("error:", error_str)
		return None

	if response.choices:
		model_response = response.choices[0].message.content
		return model_response
	else:
		return None

def get_score(response):
    match = re.search(r"(?:Score:\s*(\d)|(\d)\s*point[s]?)", response, re.IGNORECASE)
    if match:

        score = int(match.group(1) or match.group(2))
        return score
    else:
        return -1

def save_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def write_output(dataset, outputfilename):
    with open(outputfilename, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
	

def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument("--file1", type=str, help="input_file")
    parser.add_argument("--file2", type=str, help="input_file")
    parser.add_argument("--file3", type=str, help="input_file")
    parser.add_argument("--min_turns", type=int, default=5, help="minimum number of turns in a conversation")

    args = parser.parse_args()
    return args

def load_or_fallback(file):
    # session 로드 시 -1일 경우 load_data로 대체
    share_data = load_session(file)

    if share_data == -1:
        share_data = load_data(file)

    return share_data

def get_valid_score(prompt_func, *dialogues):
    # repeat until a valid score is obtained
    prompt = prompt_func(*dialogues)
    while True:
        response = get_gpt_response(prompt)
        score = get_score(response)
        if score != -1:
            return score

def calculate_average(scores_list):
    # calculate the average of the scores
    return sum(scores_list) / len(scores_list) if scores_list else 0


def main():
    args = parse_args()
    test_file = "../../datasets/test_SHARE.json"
    test_data = load_data(test_file)
    # 파일을 각각 로드 및 fallback 처리
    data1 = load_or_fallback(args.file1)
    data2 = load_or_fallback(args.file2)
    data3 = load_or_fallback(args.file3)

    total_score = 0
    minus_count = 0
    count = 0

    all_data = []

    for key, value in test_data.items():
    
        conversation = value['dialogue']
        if len(conversation) < args.min_turns:
            continue

        dialogues1 = ""
        dialogues2 = ""
        
        for dia in conversation[0]['dialogues']:
            dialogues1 += f"{dia['speaker']}: {dia['text']}\n"
        for dia in conversation[1]['dialogues']:
            dialogues2 += f"{dia['speaker']}: {dia['text']}\n"
        
        if key not in data1 or key not in data2 or key not in data3:
            break
        dialogues3 = data1[key]['dia_no_tag_text']
        dialogues4 = data2[key]['dia_no_tag_text']
        dialogues5 = data3[key]['dia_no_tag_text']

        
        # 각 점수 평가
        s_consistency_score = get_valid_score(consistency, dialogues1, dialogues2, dialogues3, dialogues4, dialogues5)
        s_reflectiveness_score = get_valid_score(reflectiveness, dialogues1, dialogues2, dialogues3, dialogues4, dialogues5)
        s_engagingness_score = get_valid_score(engagingness, dialogues1, dialogues2, dialogues3, dialogues4, dialogues5)
        

        # 각 대화에 대한 정보를 저장
        conversation_data = {
            'key': key,
            'dialogues': {
                'dialogues1': dialogues1,
                'dialogues2': dialogues2,
                'dialogues3_gen': dialogues3,
                'dialogues4_gen': dialogues4,
                'dialogues5_gen': dialogues5,
            },
            'scores': {
                'score': {
                    'consistency': s_consistency_score,
                    'reflectiveness': s_reflectiveness_score,
                    'engagingness': s_engagingness_score
                },
            }
        }
        print(f"No. {count}")
        print(conversation_data['scores'])

        all_data.append(conversation_data)
        count += 1

    # print score
    total_consistency_score = calculate_average([conversation['scores']['score']['consistency'] for conversation in all_data])
    total_reflectiveness_score = calculate_average([conversation['scores']['score']['reflectiveness'] for conversation in all_data])
    total_engagingness_score = calculate_average([conversation['scores']['score']['engagingness'] for conversation in all_data])
    # Average score

    print(f"Consistency, {total_consistency_score}")
    print(f"Reflectiveness , {total_reflectiveness_score}")
    print(f"Engagingness, {total_engagingness_score}")    
    
    print(f"file1 path: {args.file1}")
    save_json(all_data, 'conversation_scores_independent.json')



if __name__ == "__main__":
    main()