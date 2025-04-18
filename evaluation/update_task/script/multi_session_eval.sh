# METHOD= accumulate, update, noshare, independent
# BACKBONE= gemma, llama

BACKBONE='gemma'
METHOD='independent'
DATE='0216'
total_session_number=6

CUDA_VISIBLE_DEVICES=2 python ../main.py \
--method=$METHOD \
--backbone=$BACKBONE \
--input_file='../../../datasets/test_SHARE.json' \
--output_file="../results/${METHOD}_${BACKBONE}_${DATE}" \
--startnumber=0 \
--endnumber=10000000000 \
--total_session_number=$total_session_number \