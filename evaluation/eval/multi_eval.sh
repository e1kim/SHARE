


EVAL_LIST="engagingness closeness reflectiveness coherence"

for session_num in 4 5 6
do
    INPUTFILE="update_gemma_${session_num}_session.json"
    OUTPUTFILE="update_gemma_${session_num}_session"

    for EVAL in $EVAL_LIST
    do
        echo "Evaluating $EVAL for $session_num sessions"
        python multisession_evaluation.py  --input_file $INPUTFILE --output_file ${OUTPUTFILE}_${EVAL}.json --evaluation $EVAL
    done
done
