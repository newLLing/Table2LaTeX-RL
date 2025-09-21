<div align="center">

# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models

</div>
## ⚡ Updates
* 18/09/2025: 🔥 Table2LaTeX-RL has been accepted to NeurIPS 2025.

## 🚀 TL;DR
<p align="center">
  <img src="./assets/Model.png" width=100%/>
</p>

**Table2LaTeX-RL** is a reinforced multimodal LLM framework that optimizes both structural and visual rewards to accurately generate LaTeX code from complex table images.

**Key contributions of our work**:

Problem insight: We address the challenging, practical table-image-to-LaTeX conversion task and analyze its unique difficulties in depth.

MLLM-based pipeline & insight: We integrate large-scale SFT with RL, achieving strong in- and out-of-domain performance, and reveal that MLLMs can excel on tasks with substantial domain gaps from typical vision-to-language problems when paired with targeted fine-tuning and RL.

Novel RL strategy — VSGRPO: Our Visual–Structural Guided RL jointly optimizes visual quality and structural fidelity via two complementary rewards, going beyond rule-based textual correctness checks and making it well-suited for layout-and-content sensitive tasks.