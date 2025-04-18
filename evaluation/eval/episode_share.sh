SHARE1="update_NUM5_1008_llama_3_session.json"
SHARE2="update_NUM5_1008_llama_4_session.json"
SHARE3="update_NUM5_1008_llama_5_session.json"
NOSHARE1="noshare_NUM5_1013_llama_3_session.json"
NOSHARE2="noshare_NUM5_1013_llama_4_session.json"
NOSHARE3="noshare_NUM5_1013_llama_5_session.json"
SHARE1_GEMMA="update_1013_gemma_3_session.json"
SHARE2_GEMMA="update_1013_gemma_4_session.json"
SHARE3_GEMMA="update_1013_gemma_5_session.json"
NOSHARE1_GEMMA="noshare_NUM5_gemma_3_session.json"
NOSHARE2_GEMMA="noshare_NUM5_gemma_4_session.json"
NOSHARE3_GEMMA="noshare_NUM5_gemma_5_session.json"

min_turns=5

# if you want to compare two methods, you can use the following command
# python episode_evaluation.py  --share_file1 $SHARE1 --share_file2 $SHARE2 --share_file3 $SHARE3 \
#  --noshare_file1 $NOSHARE1 --noshare_file2 $NOSHARE2 --noshare_file3 $NOSHARE3

# or if you want to compare just one method, you can use the following command
python episode_evaluation_one_file.py --file1 $NOSHARE1_GEMMA --file2 $NOSHARE2_GEMMA --file3 $NOSHARE3_GEMMA --min_turns $min_turns