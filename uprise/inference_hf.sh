#export TASK=$1 # task name for evaluation, should be the same as the name in the task.py file
export TASK=openbookqa
export LLM="EleutherAI/gpt-neo-2.7B" # LLM for inference
export CACHE_DIR=$PWD/cache # directory path for caching the LLM checkpoints, task datasets, etc.
export RETRIEVER=$PWD/downloads/retriever_ckpt.2 # path to the downloaded retriever checkpoint
export PROMPT_POOL=$PWD/downloads/prompt_pool.json # path to the downloaded prompt pool
export NUM_PROMPTS=1 #original setting of 3

# retrieve prompts for each task example
# the retrieved prompts will be in '$PWD/my_data/experiment/uprise/${TASK}_prompts.json'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python DPR/dense_retriever.py \
# 	 model_file=${RETRIEVER} \
# 	 qa_dataset=qa_uprise ctx_datatsets=[dpr_uprise] \
# 	 encoded_ctx_files=[$PWD/my_data/experiment/uprise/dpr_enc_index_*] \
# 	 out_file=$PWD/my_data/experiment/uprise/${TASK}_prompts_${NUM_PROMPTS}shot.json \
# 	 datasets.qa_uprise.task_name=${TASK} \
# 	 datasets.qa_uprise.cache_dir=${CACHE_DIR} \
# 	 n_docs=${NUM_PROMPTS} \
# 	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
# 	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
# 	 encoder.cache_dir=${CACHE_DIR} \
# 	 hydra.run.dir=$PWD/my_data/experiment/uprise

# CUDA_VISIBLE_DEVICES=0,1,2,3 python DPR/dense_retriever.py \
# 	 model_file=${RETRIEVER} \
# 	 qa_dataset=qa_uprise ctx_datatsets=[dpr_uprise] \
# 	 encoded_ctx_files=[$PWD/my_data/experiment/uprise/dpr_enc_index_*] \
# 	 out_file=$PWD/my_data/experiment/uprise/${TASK}_prompts_3shot.json \
# 	 datasets.qa_uprise.task_name=${TASK} \
# 	 datasets.qa_uprise.cache_dir=${CACHE_DIR} \
# 	 n_docs=3 \
# 	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
# 	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
# 	 encoder.cache_dir=${CACHE_DIR} \
# 	 hydra.run.dir=$PWD/my_data/experiment/uprise

echo "===============================================================\
===============================================================\
===============================================================\
==============================================================="

# run vanill zero-shot baseline, 
# # the LLM predictions will be in '$PWD/my_data/experiment/uprise/${TASK}_0Shot_pred.json'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 --main_process_port \
	 23548      inference.py \
	 prompt_file=$PWD/my_data/experiment/uprise/${TASK}_prompts_1shot.json \
	 task_name=${TASK} \
	 output_file=$PWD/my_data/experiment/uprise/${TASK}_0Shot_pred.json \
	 res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
	 model_name=${LLM} cache_dir=${CACHE_DIR} \
	 num_prompts=0 batch_size=8 \
	 hydra.run.dir=$PWD/my_data/experiment/uprise

# # run UPRISE
# # the LLM predictions will be in '$PWD/my_data/experiment/uprise/${TASK}_Uprise_pred_${NUM_PROMPTS}shot.json'
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 --main_process_port \
# 	 23548       inference.py \
# 	 prompt_file=$PWD/my_data/experiment/uprise/${TASK}_prompts_${NUM_PROMPTS}shot.json \
# 	 task_name=${TASK} \
# 	 output_file=$PWD/my_data/experiment/uprise/${TASK}_Uprise_pred_${NUM_PROMPTS}shot.json \
# 	 res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
# 	 model_name=${LLM} cache_dir=${CACHE_DIR} \
# 	 num_prompts=${NUM_PROMPTS}  batch_size=4 \
# 	 hydra.run.dir=$PWD/my_data/experiment/uprise

# CUDA_VISIBLE_DEVICES=0,1,2,3  accelerate launch --num_processes 1 --main_process_port \
# 	 23548       inference.py \
# 	 prompt_file=$PWD/my_data/experiment/uprise/${TASK}_prompts_3shot.json \
# 	 task_name=${TASK} \
# 	 output_file=$PWD/my_data/experiment/uprise/${TASK}_Uprise_pred_3shot.json \
# 	 res_file=$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt \
# 	 model_name=${LLM} cache_dir=${CACHE_DIR} \
# 	 num_prompts=3  batch_size=4 \
# 	 hydra.run.dir=$PWD/my_data/experiment/uprise


# the vanilla zero-shot and UPRISE evaluation results are in '$PWD/my_data/experiment/uprise/${TASK}_evaluation_res.txt'