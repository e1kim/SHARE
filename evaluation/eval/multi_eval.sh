INPUTFILE="update_0927_6_session.json"
OUTPUTFILE="update_0927_6_test.json"

EVAL="reflectiveness"

python multisession_evaluation.py  --input_file $INPUTFILE --output_file $OUTPUTFILE --evaluation $EVAL

EVAL="engagingness"

python multisession_evaluation.py  --input_file $OUTPUTFILE --output_file $OUTPUTFILE --evaluation $EVAL

EVAL="closeness"

python multisession_evaluation.py  --input_file $OUTPUTFILE --output_file $OUTPUTFILE --evaluation $EVAL

