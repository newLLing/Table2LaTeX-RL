<div align="center">

# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

</div>

## ⚡ Updates
* 22/09/2025: 🤖 We release our model and dataset.
* 21/09/2025: 🎉 We release our codebase.
* 18/09/2025: 🔥 Table2LaTeX-RL has been accepted to NeurIPS 2025.

## 🚀 TL;DR
<p align="center">
  <img src="./assets/Model.png" width=100%/>
</p>

Complex tables in scientific papers have always been difficult to automatically reproduce. **Table2LaTeX-RL** addresses this bottleneck by proposing the first visual + structural dual-reward reinforcement learning framework (VSGRPO), enabling large models to both generate correct LaTeX code and highly reproduce the final typesetting effect.

🎯**Key contributions of our work**:

Problem insight: We address the challenging, practical table-image-to-LaTeX conversion task and analyze its unique difficulties in depth.

MLLM-based pipeline & insight: We integrate large-scale SFT with RL, achieving strong in- and out-of-domain performance, and reveal that MLLMs can excel on tasks with substantial domain gaps from typical vision-to-language problems when paired with targeted fine-tuning and RL.

Novel RL strategy — VSGRPO: Our Visual–Structural Guided RL jointly optimizes visual quality and structural fidelity via two complementary rewards, going beyond rule-based textual correctness checks and making it well-suited for layout-and-content sensitive tasks.

## 🛠️ Usage
### (Step1) Install && Training

We recommend using Docker for code training. The configuration steps are as follows:
```bash
1. Pull the docker image. 
[Docker Image]
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1

2. Run the docker image.
docker run --gpus all --ipc=host --network=host --name swift-grpo \
-v /home/code:/code \
-v /home/Datasets:/data \
-it [modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1] bash

3. Install the required dependencies. 
pip install -U trl && pip install table_recognition_metric && pip install -e .

4. Run the code. 
sh examples/train/grpo/plugin/run_external_rm.sh
```


### (Step2) Evaluation

Before running the evaluation, please download the evaluation datasets from [🤗 Table2LaTeX-RL Evaluation](https://huggingface.co/datasets/LLLHHH/Table2LaTeX-RL), and the model form [🤗 Table2LaTeX-RL Model](https://huggingface.co/LLLHHH/Table2Latex-RL)



```bash
pythoh qwenvl_test.py
```

### (Step3) Metrics
Use cw_ssim.ipynb to measure the CW-SSIM and TEDS-Structure metrics.

## Citation
If you find our works useful for your research, please consider citing:
```bibtex
@misc{ling2025table2latexrlhighfidelitylatexcode,
      title={Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models}, 
      author={Jun Ling and Yao Qi and Tao Huang and Shibo Zhou and Yanqin Huang and Jiang Yang and Ziqi Song and Ying Zhou and Yang Yang and Heng Tao Shen and Peng Wang},
      year={2025},
      eprint={2509.17589},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.17589}, 
}
```

## Acknowledgement
* This work is supported by [Research Center for Scientific Data Hub, Zhejiang Lab, Hangzhou, China](https://www.zhejianglab.org/lab/home) for computing resources.
* The training codes is built on [ms-swift](https://github.com/modelscope/ms-swift).
* The base model is from [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
