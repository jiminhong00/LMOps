#!/bin/bash

export RETRIEVER=$PWD/downloads/retriever_ckpt.2 # path to the downloaded retriever checkpoint
export PROMPT_POOL=$PWD/downloads/prompt_pool.json # path to the downloaded prompt pool
export CACHE_DIR=$PWD/cache # directory path for caching the LLM checkpoints, task datasets, etc.

rm -r cache/*

CUDA_VISIBLE_DEVICES=$1 python DPR/generate_dense_embeddings.py \
	 model_file=${RETRIEVER} \
	 ctx_src=dpr_uprise shard_id=0 num_shards=1 \
	 out_file=$PWD/my_data/experiment/uprise/dpr_enc_index \
	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
	 encoder.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PWD/my_data/experiment/uprise


	 # noproblem with gpu 0123 // 4567 // 2345 // 07 // 06 // 16 // 17 // 016 // 017 // 067 // 167
	 # noproblem 2 w/ gpu 1267
	 # problem with all / 0167
	 # noproblem ver 3 w/ gpu 0167


	 # p all / 
	 # np 0123 / 4567 // 2345 // 0167

	 #몰라 그냥 하자. 