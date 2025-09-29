# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 80GiB
export NCCL_TIMEOUT=3600000

CUDA_VISIBLE_DEVICES=2,3 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --use_vllm true \
    --vllm_enforce_eager true \
    --vllm_gpu_memory_utilization 0.7 \
    --model /mnt/geogpt-doc/docparser/uestc/ms-swift/qwenvl2.5-3b-1epoch \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_table2latex_acc external_table2latex_form \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /mnt/geogpt-doc/docparser/uestc/InternVL/complex/table2latex.jsonl \
    --max_completion_length 2000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 2 \
    --temperature 0.9 \
    --log_completions false \
    --freeze_vit true \
    --freeze_parameters visual \
    --split_dataset_ratio 0 \
    --deepspeed zero3 \
    --num_infer_workers 2 \
    --use_liger_kernel true \
    --max_pixels 524288 \


