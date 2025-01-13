export CUDA_VISIBLE_DEVICES=2 # 0,1 

TASK_NAME='w_tag' #wo_tag, w_tag, baseline
MODEL_NAME='base_llama' #llama, gemma, base_llama, base_gemma
DATA_PATH=''  #evaluation dataset path
MODEL_TAG='gold_tag' #model_tag, gold_tag

python evaluation_automatic.py --task_name $TASK_NAME --model_name $MODEL_NAME --data_path $DATA_PATH --model_tag $MODEL_TAG
