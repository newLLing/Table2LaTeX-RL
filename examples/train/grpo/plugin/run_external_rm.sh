export NCCL_TIMEOUT=3600000

torchrun \
    --nproc_per_node=8 \
    --nnodes=${WORLD_SIZE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank=${RANK} \
    swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model Qwen2.5-VL-3B \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_table2latex_acc external_table2latex_form \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/geogpt-doc/docparser/data/table/v1.4/table2latex_grpo_hard_5936_swift.jsonl \
    --max_completion_length 3000 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy epoch \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir output_sft_grpo_complex_acc0.4 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --log_completions false \
    --freeze_vit true \
    --freeze_parameters visual \
    --split_dataset_ratio 0 \
    --deepspeed zero3 \
    --save_only_model true \
    --max_pixels 524288