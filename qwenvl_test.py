import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import datetime

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    timeout = dist.default_pg_timeout = 3600000.0
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout)
    )
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def resize_image(image_path, max_size=1333):
    """调整图像尺寸，最长边不超过max_size，保持宽高比"""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def process_batch(batch_items, model, processor, device):
    batch_images = []
    batch_texts = []
    
    for item in batch_items:
        conv = item['conversations'][0]
        human_msg = conv['value']
        question = human_msg.split('<image>\n')[-1].replace('<image>', '').strip() + " Please output only the LaTeX code between \\begin{tabular} and \\end{tabular}, without any additional text or explanations."
        
        # 加载并调整图像尺寸
        resized_image = resize_image(item['image'])
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": question}
            ]
        }]
        
        image_inputs, _ = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        batch_images.append(image_inputs[0])
        batch_texts.append(text)
    
    inputs = processor(
        text=batch_texts,
        images=batch_images,
        padding=True,
        return_tensors="pt",
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.module.generate(
            **inputs,
            max_new_tokens=8192,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=False
        )
    
    return processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

def process_on_device(rank, world_size, input_path, output_path, batch_size):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        # 初始化模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/data/shared/Qwen/LLaMA-Factory/qwen/qwenvl-2.5-grpo-5epoch",
            torch_dtype=torch.float16
        )
        model = model.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank)
        model.eval()
        
        processor = AutoProcessor.from_pretrained(
            "/data/shared/Qwen/LLaMA-Factory/qwen/qwenvl-2.5-grpo-5epoch"
        )
        
        # 加载并分片数据
        with open(input_path, 'r') as f:
            dataset = [json.loads(line) for line in f]
        
        per_device_data = dataset[rank::world_size]
        results = []
        
        # 处理数据
        with tqdm(total=len(per_device_data), desc=f"GPU {rank}") as pbar:
            for i in range(0, len(per_device_data), batch_size):
                batch_items = per_device_data[i:i+batch_size]
                decoded_texts = process_batch(batch_items, model, processor, device)
                
                for item, generated_text in zip(batch_items, decoded_texts):
                    results.append({
                        "id": item["id"],
                        "image": item["image"],
                        "question": item['conversations'][0]['value'],
                        "prediction": generated_text.strip(),
                        "reference": item['conversations'][1]['value']
                    })
                
                pbar.update(len(batch_items))
        
        # 同步并收集结果
        gathered_results = [None for _ in range(world_size)]
        dist.barrier()  # 确保所有进程都完成处理
        dist.all_gather_object(gathered_results, results)
        
        if rank == 0:
            # 合并并保存结果
            final_results = []
            for r in gathered_results:
                final_results.extend(r)
                
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in final_results:
                    f.write(json.dumps(result) + '\n')
    
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    input_path = "/data/shared/Qwen/InternVL/internvl_chat/dataset/complex/table2latex.jsonl"
    output_path = "/data/shared/Qwen/LLaMA-Factory/output/results_qwenvl3b_complex_just_grpo_5epoch/output.jsonl"
    
    # 每个GPU的batch size
    per_gpu_batch_size = 1
    
    mp.spawn(
        process_on_device,
        args=(world_size, input_path, output_path, per_gpu_batch_size),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()