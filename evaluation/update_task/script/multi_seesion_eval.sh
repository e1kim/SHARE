# METHOD= accumulate, update, noshare
# BACKBONE= gemma, llama

BACKBONE='llama'
METHOD='notag'
DATE='1013'
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--method=$METHOD \
--backbone=$BACKBONE \
--input_file='../../../datasets/SHARE.json' \
--output_file="../results/${METHOD}_${BACKBONE}_${DATE}" \
--startnumber=0 \
--endnumber=4 \
--total_session_number=5