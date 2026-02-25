# MERGETUNE: Continued Fine-tuning of Vision-Language Models [ICLR 2026]

**[MergeTune: Continued Fine-Tuning of Vision-Language Models](https://arxiv.org/pdf/2601.10497)**

**Wenqing Wang¹\*, Da Li²³\*, Xiatian Zhu¹†, Josef Kittler¹†**

¹ University of Surrey · ² Samsung AI Centre Cambridge · ³ Queen Mary University of London

\* Equal contribution · † Joint last authorship

---

## Abstract

Fine-tuning vision-language models (VLMs) such as CLIP often leads to catastrophic forgetting of pretrained knowledge. Prior work primarily aims to mitigate forgetting during adaptation; however, forgetting often remains inevitable during this process. We introduce a novel paradigm, **continued fine-tuning (CFT)**, which seeks to recover pretrained knowledge after a zero-shot model has already been adapted. We propose a simple, model-agnostic CFT strategy (named **MERGETUNE**) guided by linear mode connectivity (LMC), which can be applied post hoc to existing fine-tuned models without requiring architectural changes. Given a fine-tuned model, we continue fine-tuning its trainable parameters (e.g., soft prompts or linear heads) to search for a continued model which has two low-loss paths to the zero-shot (e.g., CLIP) and the fine-tuned (e.g., CoOp) solutions. By exploiting the geometry of the loss landscape, the continued model implicitly merges the two solutions, restoring pretrained knowledge lost in the fine-tuned counterpart. A challenge is that the vanilla LMC constraint requires data replay from the pretraining task. We approximate this constraint for the zero-shot model via a second-order surrogate, eliminating the need for large-scale data replay.

## Method Overview

<p align="center">
  <a href="assets/pipeline.pdf">**Figure 2: The proposed MERGETUNE (conceptual illustration)** — View high-quality PDF</a>
</p>

*Figure 2: The proposed MERGETUNE (conceptual illustration).* (Left) Before MERGETUNE Training: The zero-shot model ŵ₁ and fine-tuned ŵ₂ exist in separate minima with no low-loss connectivity. Linear interpolation between them (shown in the inset) reveals high barriers and induces a performance trade-off on base and novel classes. (Middle) During training, *w* is searched to be mode-connected to both ŵ₁ and ŵ₂, gradually integrating both models. (Right) After MERGETUNE Training: Our continued model *w_ours* merging two endpoints will be used for inference of both tasks ŵ₁ and ŵ₂ where trained. The two distinct low-loss paths, ŵ₁ → *w_ours* and ŵ₂ → *w_ours*, show smooth interpolation curves (inset) indicating stable performance.
---


## Setup

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended)

### Installation

**Step 1: Create conda environment and install PyTorch**

```bash
# Create a conda environment
conda create -n mergetune python=3.7
conda activate mergetune

# Install PyTorch (>= 1.8.1) and torchvision
# See https://pytorch.org/ for other CUDA versions
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

**Step 2: Install Dassl.ProGrad.pytorch**

```bash
cd Dassl.ProGrad.pytorch
pip install -r requirements.txt
pip install -e .
cd ..
```

**Step 3: Install MERGETUNE dependencies**

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```



## Datasets

For dataset preparation and setup instructions, please refer to the [CoOp DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) or other CoOp/CLIP-based repositories, as we follow the same data organization.




## Usage

MERGETUNE follows a two-stage pipeline: (1) train the base model to obtain checkpoints, then (2) apply MERGETUNE to continue fine-tuning with LMC.

### Training

**CoOp + MERGETUNE**

```bash
cd mergetune
# Stage 1: Train CoOp
bash scripts/coop/base2new_train.sh imagenet 1  # dataset seed

# Stage 2: Apply MERGETUNE (requires CoOp checkpoint)
bash scripts/coop_LMC/base2new_train_coop.sh imagenet 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
# args: dataset clip_weight loss_type coop_lmc w_lmc seed config
```

**KgCoOp + MERGETUNE**

```bash
# Stage 1: Train KgCoOp
bash scripts/kgcoop/base2new_train.sh imagenet 8.0 1  # dataset weight seed

# Stage 2: Apply MERGETUNE
bash scripts/kgcoop_LMC/base2new_train_kgcoop.sh imagenet 10.0 1.0 1 vit_b16_ep100_ctxv1
# args: dataset w w_lmc seed config
```

**MMA + MERGETUNE**

```bash
# Stage 1: Train MMA
bash scripts/mma/base2new_train.sh imagenet 1 vit_b16_ep5  # dataset seed config

# Stage 2: Apply MERGETUNE
bash scripts/mma_LMC/base2new_train_mma.sh imagenet 10.0 vit_b16_ep5 1
# args: dataset w_lmc config seed
```

**PromptKD + MERGETUNE**

```bash
# Stage 1: Train PromptKD (requires pretrained teacher model)
# Stage 2: Apply MERGETUNE
bash scripts/promptkd_LMC/base2new_train_promptkd.sh caltech101 1 <pretrained_student_path> 10.0
# args: dataset seed pretrained_student_path kd_weight
```

For more examples, see [train.sh](mergetune/train.sh).

### Evaluation

After training, run evaluation scripts with the same arguments as training:

```bash
# CoOp + MERGETUNE
bash scripts/coop/base2new_test.sh imagenet 1
bash scripts/coop/base2base_test.sh imagenet 1
bash scripts/coop_LMC/base2new_test_coop.sh imagenet 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
bash scripts/coop_LMC/base2base_test_coop.sh imagenet 10.0 cosine True 1.0 1 vit_b16_ep100_ctxv1
```

Replace `imagenet` with your dataset (e.g., `caltech101`, `oxford_pets`) and match the arguments to your training run. See [train.sh](mergetune/train.sh) for full evaluation examples.


## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2024mergetune,
  title={MergeTune: Continued Fine-Tuning of Vision-Language Models},
  author={Wang, Wenqing and Li, Da and Zhu, Xiatian and Kittler, Josef},
  journal={arXiv preprint arXiv:2601.10497},
  year={2024}
}
```

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [CoOp](https://github.com/KaiyangZhou/CoOp) by Kaiyang Zhou
- [KgCoOp](https://github.com/htyao89/KgCoOp) by Hantao Yao et al.
- [MMA](https://github.com/ZjjConan/VLM-MultiModalAdapter) by Lingxiao Yang et al.
- [PromptKD](https://github.com/zhengli97/PromptKD) by Zheng Li et al.
- [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) framework


## 📧 Contact

For questions about the code or paper, please contact [wenqing.wang@surrey.ac.uk].
