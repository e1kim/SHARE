# METHOD= accumulate, update, noshare, independent
# BACKBONE= gemma, llama

BACKBONE='llama'
METHOD='independent'
DATE='0127'
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--method=$METHOD \
--backbone=$BACKBONE \
--input_file='../../../datasets/test_SHARE.json' \
--output_file="../results/${METHOD}_${BACKBONE}_${DATE}" \
--startnumber=0 \
--endnumber=10000000 \
--total_session_number=5