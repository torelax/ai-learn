# data example:
# {"query": str, "pos": List[str], "neg":List[str], "prompt": str}

# ```shell
# torchrun --nproc_per_node {number of gpus} \
# -m FlagEmbedding.llm_reranker.finetune_for_instruction.run \
# --output_dir {path to save model} \
# --model_name_or_path google/gemma-2b \
# --train_data ./toy_finetune_data.jsonl \
# --learning_rate 2e-4 \
# --num_train_epochs 1 \
# --per_device_train_batch_size 1 \
# --gradient_accumulation_steps 16 \
# --dataloader_drop_last True \
# --query_max_len 512 \
# --passage_max_len 512 \
# --train_group_size 16 \
# --logging_steps 1 \
# --save_steps 2000 \
# --save_total_limit 50 \
# --ddp_find_unused_parameters False \
# --gradient_checkpointing \
# --deepspeed stage1.json \
# --warmup_ratio 0.1 \
# --bf16 \
# --use_lora True \
# --lora_rank 32 \
# --lora_alpha 64 \
# --use_flash_attn True \
# --target_modules q_proj k_proj v_proj o_proj

# ```