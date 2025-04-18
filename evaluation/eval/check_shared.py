# open json file
import json
import os

import argparse

noshared_couple = """
('EDDIE', 'ROSE')
('CRONIN', 'PAMELA')
('MARTIN', 'SHERMAN')
('MO', 'SHADES')
('HERNANDEZ', 'TAFT')
('LATESHA', 'SCOTT')
('JOHN', 'RON')
('ALEC', 'LESLIE')
('EADY', 'NEIL')
('CONNELL', 'JAMES')
('CORNWALLIS', 'TARLETON')
('NAPOLEON (V.O.)', 'NARRATOR')
('LAROCHE', 'ORLEAN')
('CATHY', 'RAYMOND')
('RENNIE', 'SEAN')
('HENRY', 'JIM')
('CHANG', 'CONWAY')
('AGENT MARIA HILL', 'NICK FURY')
('EMILIE', 'SCHINDLER')
('JENNY', 'TEENAGER')
('BENNY', 'SCOTTY')
('BOWMAN', 'SANTEN')
('KRAMER', 'McCROSKEY')
('ERIN', 'PAULSEN')
('DIRECTOR', 'WELLES')
('ARTIE', 'POLLY')
('MICHELLE', 'PATRICK')
('ALLY', 'TYLER')
('GREEN', 'KENNY')
('ELSIE', 'POLLY')
('HELEN', 'JENNY')
('AGENT PHIL COULSON', 'NICK FURY')
('D.S.', 'DREW')
('JOHN', 'YURI')
('BEN', 'ELENA')
('ROXANNE', 'WILLARD')
('KURT', 'SEEBAND')
('COP #1', 'COP #2')
('KAREN', 'WARDEN')
('D.J.', 'FERRIS')
('NICK', 'RAYMOND')
('ERIC', 'SKULL COWBOY')
('BUD', 'COBB V.O.')
('BLANCA', 'MARIA')
('EMMET', 'KING KARL')
('BIZZLEBEK', 'KAFKA')
('DAVID', 'WOMAN')
('DAN', 'MARTIN')
('HELEN', 'KLAATU')
('CLERK', 'MOSS')
('JULIUS', 'WEBSTER')
('KICHIJIRO', 'RODRIGUES')
('CLARICE', 'MR. GUMB')
('CHRISTY', 'MOTHER SUPERIOR')
('FRANK', 'JIM')
('RICHARD', 'SAM')
('NEO', 'TANK')
('GORDON', 'RITA')
('JENNIFER', 'TOM')
('KELLY', 'KEOUGH')
('CARLOS', 'EVAN')
('BLAIREAU', 'VICTOR')
('HARRY', 'MARY')
('MATT', 'SARA')
('FISK', 'JACK')
('BOWER', 'NADIA')
('NORMAN', 'WARDADDY')
('ADELE', 'CARRIE')
('DIRK', 'REED')
('A.J.', 'LEV')
('CHIRON', 'KEVIN')
('DERM', 'ED')
('AMOS', 'RAMIREZ')
('BLAIREAU', 'MATRE GUILLOCHE')
('POPS', 'SPRITTLE')
('CAROLINE', 'VICTOR')
('BLANC', 'RANSOM')
('HELEN', 'M.J.')
('CORKY (V.O.)', 'VIOLET (V.O.)')
('COCO', 'HELMUT')
('BORG QUEEN', 'DATA')
('ADELE', 'BRIAN')
('SARAH', 'ZAKAR')
('EVELYN', 'RAFE')
('BUCKY', 'KOENIG')
('RED', 'SALLY')
('DRIVER', 'PRETTY TEENAGE GIRL')
('ISHMAEL', 'STARBUCK')
('BROUSSARD', 'STANDARD')
('JEFF', 'KALE')
('CARY', 'JOE')
('JO', 'KAFFEE')
('ANWAR', 'RICHARD')
('PETER', 'SHERMAN')
("SA'ID", 'TROY')
('LOVELESS', 'PRESIDENT GRANT')
('CASALS (V.O.)', 'HANNA')
('NIMZIKI', 'PRESIDENT')
('HOWARD', 'MARY')
('ALI', 'MARCUS')
('PERRY BABCOCK', 'SANDRA BABCOCK')
('MITCHELL', 'SEAN')
('HICKS', 'REYNOLDS')
('MILLER', 'SMITH (O.S.)')
('LEONARD', 'PAULA')
('BUD', 'CATFISH')
('HERRICK', 'ROWENA')
('BRODY', 'HOOPER')
('MALCOLM', 'NICK')
('ENGLEHORN', 'HAYES')
('ABBIE', 'JAMIE')
('MICHAEL', 'PRIEST')
('SOFYA', 'VALENTIN')
('BRADY', 'WAYNE')
('BEVERLY', 'RACINE')
('JIMMY', 'JUDY (O.C.)')
('OLIVER', 'ZED')
('HARDY', 'LUCILLE')
('IRENE', 'MIRANDA')
('BREAKER', 'SCARLETT')
('KIRK', 'SAAVIK')
('RICK', 'STANLEY')
('PAUL', 'TOM')
('JAKE', 'TONY')
('CELIA', 'EVA')
('MUMFORD (V.O.)', 'SKIP (V.O.)')
('EARL', 'PHIL')
('KAUFMAN', 'MIKE OWEN')
('LINDA', 'LUCY')
('XANDER', 'YORGI')
('ADAM', 'CLERK')
('LANDON', 'REV. SULLIVAN')
"""

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def parse_args():
	parser = argparse.ArgumentParser(description="take Model")
	parser.add_argument("--input_file", type=str, help="input_file")
	parser.add_argument("--output_file", type=str, help="output_file")
	parser.add_argument("--evaluation", type=str, help="evaluation")

	args = parser.parse_args()
	return args

def main():
    args = parse_args()
    noshared_eval_list = []
    shared_eval_list = []

    for file_num in ["4", "5", "6"]:
        
        for eval in ["engagingness", "coherence", "reflectiveness", "closeness"]:
            file_path = f"accum_0927_{file_num}_session_{eval}.json"
   
            dataset = read_json_file(file_path)
            no_shared_score = 0
            shared_score = 0
            total = 0

            for key,value in dataset.items():
                if key in noshared_couple:
                    no_shared_score += value[f"{eval}"]
                    total += 1
                else:
                    shared_score += value[f"{eval}"]

            noshared_eval_list.extend([no_shared_score/total])
            shared_eval_list.extend([shared_score/(96-total)])

        
    for csv in noshared_eval_list:
        print(f"{csv},", end="")
    print()

    for csv2 in shared_eval_list:
        print(f"{csv2},", end="")
    print()



if __name__ == '__main__':

    main()