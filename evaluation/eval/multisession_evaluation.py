from openai import OpenAI
import os, sys
import json
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
import argparse
import re


def engagingness(dialogue):
    text=f"""\
You will be given a conversation between two participants. Your task is to read, understand, and evaluate the interaction based on how it subtly reveals emotional complexity, relational depth, and captivating dynamics. The conversation should reflect moments of emotional engagement, relational tension, or even quiet significance that makes the interaction feel engaging and memorable.

Engagingness Evaluation Criteria:

Engagingness measures how much the conversation entertains through emotional layers, subtle relational tension, or significant moments that might not immediately appear dramatic but still add depth. The conversation should feel dynamic and engaging, with interactions that reveal or suggest deeper emotional connections or personal significance, even in seemingly calm exchanges. It's important to consider how elements of trust, vulnerability, or tension might quietly enrich the dialogue, making it captivating and entertaining.

Key Aspects to Consider:

Subtle Emotional Depth and Relational Tension: The conversation should contain moments where emotional layers or relational tension are hinted at, even if not overt. Subtle expressions of humor, trust, or tension can add richness to the interaction, enhancing engagement through underlying dynamics between participants.

Significant Moments, Even in Calm Exchanges: The conversation may include moments that seem quiet or routine on the surface but carry significant emotional weight or meaning. These moments should be considered memorable, as they contribute to the deeper dynamics of the interaction, leaving a lasting impression.

Consistent Engagement and Entertainment Value: The conversation should create a flow that keeps the reader invested, revealing dynamics between participants that make the interaction entertaining, even in subtle ways. The dialogue may include calm exchanges, but it should feel engaging through the emotions and relational depth that lie beneath the surface.

Scoring Guide:

0 points: The conversation lacks emotional complexity, depth, or relational engagement, making it trivial or forgettable.

1 point: The conversation has minimal emotional engagement, with few moments of relational tension or humor, but they are not sustained or impactful.

2 points: The conversation includes moments of emotional engagement or humor but is inconsistent in creating a strong impact throughout.

3 points: The conversation consistently engages and entertains through emotional layers, relational depth, or significant moments that reveal underlying dynamics. Even seemingly simple or calm exchanges carry emotional weight, making the interaction feel rich and captivating, with memorable moments that enhance the overall engagement.

Score: [score]

Now, based on the conversation below, evaluate the Engagingness score by considering how captivating, creative, and emotionally resonant the dialogue is.

Dialogue: 
{dialogue}
Score:"""
    return text

def closeness(dialogue):
    text = f"""\

You will be given a conversation between two participants. Your goal is to read, remember, and understand the dialogue to evaluate how well it reflects the participants' mutual understanding and familiarity with each other.

Closeness Evaluation Criteria:

Closeness measures how well the participants know each other, reflecting the depth of their relationship and the extent of their understanding of one another. This evaluation goes beyond mere communication, assessing how their interactions reveal a shared history, familiarity, or dynamic that shapes their connection. Whether the interaction is friendly, contentious, or competitive, the key is how well the participants recognize and respond to each other's traits, emotions, and communication styles. Elements such as light teasing, the use of slang, emotional support, sharing personal information, forming empathy, and communication flexibility are vital indicators of closeness, even in adversarial exchanges.

Key Aspects to Consider:

Depth of Mutual Recognition: Look for moments where the participants acknowledge and respond to each other's unique traits and past experiences, including light-hearted jokes or teasing, indicating familiarity and understanding.

Emotional Support and Personal Sharing: Assess how the participants provide emotional support during challenging times and share personal information, such as discussing family matters or past mistakes, which enhances their connection.

Adaptation to Each Other's Responses: Evaluate how the participants adjust their behavior or language in response to one another, showcasing their comfort and familiarity. This adaptation can reflect a complex relationship where they navigate each other's emotions effectively.

Scoring Guide:

0 points: The conversation lacks any sense of closeness. There is no mutual understanding or familiarity, making the interaction feel impersonal and disconnected.

1 point: The conversation shows minimal closeness. There are hints of mutual understanding, but the interaction remains largely superficial and lacks depth.

2 points: The conversation reflects moderate closeness. The participants demonstrate a reasonable understanding of each other through their exchanges, recognizing each other's traits, motivations, and emotional states.

3 points: The conversation demonstrates high closeness. The interaction reveals a deep familiarity and consistent recognition of each other's character, history, and dynamics. The dialogue feels like a natural extension of a long-standing relationship, reflecting profound mutual understanding, whether the context is friendly, competitive, or contentious.

Score: [score]

Now, based on the provided conversation, evaluate which response has better Coherence considering the flow, logical consistency, and emotional depth of the dialogue.

Dialogue :
{dialogue}
Score:"""
    return text

def coherence(dialogue):
	text = f"""\
You will be given a conversation between two participants. Your task is to read, remember, and understand the dialogue to evaluate how logically consistent and naturally flowing the interaction is, reflecting a coherent and human-like conversation.

Coherence Evaluation Criteria:

Coherence measures how human-like and logically consistent the conversation feels. It evaluates how well the dialogue flows naturally, maintains a clear and logical progression of ideas, and how responses are relevant to previous statements. The primary focus is on the overall narrative's engagement and emotional resonance, allowing for dynamic or intense shifts that enrich the interaction. Coherence reflects the conversation's ability to feel connected, seamless, and meaningful, resembling a natural human-to-human interaction.

Key Aspects to Consider

Overall Narrative Engagement and Logical Consistency: Responses should maintain an engaging and coherent narrative, even if some shifts feel intense or emotionally charged. Focus on the larger storyline and how well the conversation captures attention, rather than perfect alignment in each turn.

Emotional Resonance and Continuity: Emotional intensity and personal expression are valued, even if moments of tension exist. The conversation should show emotional engagement that adds depth and richness, with allowances for passionate or assertive exchanges that reflect genuine human emotions.

Dynamic Flow and Adaptability: A naturally dynamic flow with adaptability is prioritized, allowing for shifts between tension and resolution that feel real and compelling. Minor disruptions or intense reactions are acceptable if they contribute to the overall coherence and human-like quality of the dialogue.

0 points: The conversation lacks coherence. Responses are disjointed, irrelevant, or nonsensical, making the dialogue feel artificial and disconnected.

1 point: The conversation shows minimal coherence. Responses are occasionally relevant but often feel awkward or forced, disrupting the natural flow of dialogue.

2 points: The conversation is moderately coherent. Most responses make sense and follow the context, contributing to a relatively smooth and connected dialogue, even when dealing with intense or emotionally charged topics.

3 points: The conversation is highly coherent, with responses that flow naturally, maintain logical progression, and closely resemble a human-like conversation with emotional depth and continuity.

The output format should be as follows:

Score: [score]

Now, based on the provided conversation, evaluate which response has better Coherence considering the flow, logical consistency, and emotional depth of the dialogue.

Dialogue : 
{dialogue}
Score:""" 

	return text


def reflectiveness(dialogue, shared_memory):
    text = f"""\
You will be given a conversation between two participants. Your task is to read, remember, and understand the dialogue to evaluate how well it reflects the participants' shared memories, past interactions, and mutual history.

Reflectiveness Evaluation Criteria: Reflectiveness measures how well the conversation incorporates shared experiences, past events, and personal history between the participants. It evaluates how memories and mutual understanding are used to inform the interaction, reflecting a deeper connection that goes beyond surface-level dialogue. The primary focus is on the integration of past experiences and how well these memories are acknowledged, referenced, or built upon throughout the conversation.

Key Aspects to Consider:

Integration of Shared Memories: Assess how well the participants incorporate past events, memories, or experiences into the conversation. This can include direct references to shared history, implicit acknowledgments of past interactions, or the subtle use of memories that add depth and context to the dialogue.

Consistency with Past Interactions: Evaluate how consistently the conversation aligns with previously established dynamics, events, or shared experiences. The interaction should feel like a continuation of their mutual history, with responses that logically and emotionally connect to their past.

Mutual Recognition of History and Context: Look at how the participants recognize and respond to the shared context of their relationship. This includes acknowledging each other's past actions, decisions, or shared journeys, demonstrating that their connection is informed by a rich and evolving history.

Scoring Guide:

0 points: The conversation lacks any reflectiveness. There is no sense of shared history or mutual recognition of past experiences, making the dialogue feel disconnected from their relationship.

1 point: The conversation shows minimal reflectiveness. There are some hints of shared memories or past events, but the interaction lacks depth and consistency in incorporating these elements.

2 points: The conversation reflects moderate reflectiveness. Shared memories and past experiences are reasonably integrated, adding context and depth to the interaction, even if subtle.

3 points: The conversation demonstrates high reflectiveness. The interaction consistently incorporates shared history and past interactions, weaving memories into the dialogue in a meaningful and engaging way that enriches their connection.

Output Format:

Score: [score]

Now, based on the provided conversation, evaluate the Reflectiveness score by considering how well the dialogue incorporates shared memories, past events, and the mutual history between the participants.

Dialogue :
{dialogue}
Past memory :
{shared_memory}
Score:"""
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

def parse_args():
	parser = argparse.ArgumentParser(description="take Model")
	parser.add_argument("--input_file", type=str, help="input_file")
	parser.add_argument("--output_file", type=str, help="output_file")
	parser.add_argument("--evaluation", type=str, help="evaluation")

	args = parser.parse_args()
	return args


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

	# 모델의 텍스트 응답 추출
	if response.choices:
		model_response = response.choices[0].message.content
		return model_response
	else:
		return None

def get_score(response):
    # 정규식으로 "Score: 숫자" 또는 "숫자 point(s)" 형태의 숫자 부분을 추출
    match = re.search(r"(?:Score:\s*(\d)|(\d)\s*point[s]?)", response, re.IGNORECASE)
    if match:
        # 첫 번째 그룹이 매치되면 사용, 아니면 두 번째 그룹 사용
        score = int(match.group(1) or match.group(2))
        return score
    else:
        return -1

def save_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# 내용을 저장하는 함수
def save_to_txt(evaluation, input_data, total_score, minus_count, args):
	input_file = os.path.splitext(os.path.basename(args.input_file))[0]
	with open(f"update1004.txt", 'a') as file:  # 'a' 모드로 열어서 이어서 작성
		file.write(f"Evaluation: {evaluation}\n")
		file.write(f"Input data counts: {len(input_data)}\n")
		file.write(f"Average score: {(total_score/len(input_data)):0.4f}\n")
		file.write(f"Error counts: {minus_count}\n")
		file.write(f"OUTPUT FILE NAME: {args.output_file}\n")
		file.write('-'*100 + '\n')

def write_output(dataset, outputfilename):
    with open(outputfilename, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
	
def return_memory(info):
    def process_list(lst):
        if not lst:  # 리스트가 비어있으면 빈 문자열을 반환
            return ''
        return '\n'.join(['- ' + string for string in lst])

    # 정보를 처리하는 부분
    persona1 = process_list(info['information']['p1'])
    persona2 = process_list(info['information']['p2'])
    personal1 = process_list(info['information']['t1'])
    personal2 = process_list(info['information']['t2'])
    shared_memory = process_list(info['information']['shared'])
    mutual_memory = process_list(info['information']['mutual'])

    memory = f"""\
{persona1}{persona2}{personal1}{personal2}{shared_memory}{mutual_memory}"""
        
    return memory

def return_shared_memory(info):
    def process_list(lst):
        if not lst:  # 리스트가 비어있으면 빈 문자열을 반환
            return ''
        return '\n'.join(['- ' + string for string in lst])

    # 정보를 처리하는 부분
    shared_memory = process_list(info['information']['shared'])


    memory = f"""\
{shared_memory}"""
        
    return memory

def main():
	args = parse_args()

	input_data = load_session(args.input_file)

	if input_data == -1:
		input_data = load_data(args.input_file)


	total_score = 0
	minus_count = 0
	evaluation = args.evaluation
	output_file_name = args.output_file


	for key, value in input_data.items():

		dialogue = value['dia_no_tag_text']


		if evaluation == "coherence":
			prompt = coherence(dialogue)

		elif evaluation == "reflectiveness":
			previous_info = value['previous info']
			shared_memory = return_shared_memory(previous_info)
			prompt = reflectiveness(dialogue, shared_memory)

		elif evaluation == "engagingness":
			prompt = engagingness(dialogue)

		elif evaluation == "closeness":
			prompt = closeness(dialogue)
		else:
			assert False, "No evaluation"
		
		### Get GPT response and evaluate until a valid score is achieved ###
		while True:
			response = get_gpt_response(prompt)
			score = get_score(response)

			# Check if the score is valid; if -1, retry the response
			if score != -1:
				break
		
		value[args.evaluation] = score

		if score >= 0:
			total_score += score
		else:
			minus_count += 1
		
		print("Score:",response)

	
	save_to_txt(evaluation, input_data, total_score, minus_count, args)
	save_json(input_data, filename=output_file_name)


if __name__ == "__main__":
    main()