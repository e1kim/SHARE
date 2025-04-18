# METHOD= update, notag, baseline
# BACKBONE= gemma, llama

BACKBONE='llama'
METHOD='baseline'
DATE='0210'
CUDA_VISIBLE_DEVICES=2 python ../last_utter.py \
--method=$METHOD \
--backbone=$BACKBONE \
--input_file='../../../datasets/SHARE.json' \
--output_file="../results/${METHOD}_${BACKBONE}_${DATE}_last" \
--startnumber=0 \
--endnumber=4 \
--total_session_number=5