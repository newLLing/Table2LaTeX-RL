<div align="center">

# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

</div>

## ⚡ Updates
* 21/09/2025: 🎉 We release our codebase.
* 18/09/2025: 🔥 Table2LaTeX-RL has been accepted to NeurIPS 2025.

## 🚀 TL;DR
<p align="center">
  <img src="./assets/Model.png" width=100%/>
</p>

**Table2LaTeX-RL** is a reinforced multimodal LLM framework that optimizes both structural and visual rewards to accurately generate LaTeX code from complex table images.

🎯**Key contributions of our work**:

Problem insight: We address the challenging, practical table-image-to-LaTeX conversion task and analyze its unique difficulties in depth.

MLLM-based pipeline & insight: We integrate large-scale SFT with RL, achieving strong in- and out-of-domain performance, and reveal that MLLMs can excel on tasks with substantial domain gaps from typical vision-to-language problems when paired with targeted fine-tuning and RL.

Novel RL strategy — VSGRPO: Our Visual–Structural Guided RL jointly optimizes visual quality and structural fidelity via two complementary rewards, going beyond rule-based textual correctness checks and making it well-suited for layout-and-content sensitive tasks.

## 🛠️ Usage
### (Step1) Install && Training

We recommend using Docker for code training. The configuration steps are as follows:
```bash
1. Pull the docker image. [Docker Image](docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1)

2. Run the docker image.(docker run --gpus all --ipc=host --network=host --name swift-grpo \
-v /home/code:/code \
-v /home/Datasets:/data \
-it [modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.1] bash)

3. Install the required dependencies. (pip install -U trl && pip install table_recognition_metric && pip install -e .)

4. Run the code. (sh examples/train/grpo/plugin/run_external_rm.sh)
```


### (Step2) Evaluation

Before running the evaluation, please download the evaluation datasets from [🤗 Table2LaTeX-RL Evaluation](https://huggingface.co/datasets/LLLHHH/Table2LaTeX-RL), and the model form [🤗 Table2LaTeX-RL Model](https://huggingface.co/LLLHHH/Table2Latex-RL)

> 🚧 TODO:Updating model and datasets.

```bash
pythoh qwenvl_test.py
```

### (Step3) Evaluation Metrics
Use cw_ssim.ipynb to measure the CW-SSIM and TEDS-Structure metrics.

## Citation
If you find our works useful for your research, please consider citing:
```bibtex

```

## Acknowledgement
* This work is supported by [Research Center for Scientific Data Hub, Zhejiang Lab, Hangzhou, China](https://www.zhejianglab.org/lab/home) for computing resources.
* The training codes are built on [ms-swift](https://github.com/modelscope/ms-swift).
* The base models are from [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
