# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
> **Authors: Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao**


[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.00975)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Chaos96/NTPP)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/aigc-x/NTPP)
[![Replicate](https://replicate.com/ictnlp/llama-omni/badge)](https://audio-3059.pages.dev/)



<!-- <embed src="assert/audio-introduction.pdf" width="620" height="500" type="application/pdf"> -->

Key features:
- Pre-training: Transform single-channel audio into discrete tokens for next-token prediction
- SFT: Novel "next-token-pair prediction" objective for natural conversation comprehension
- Result: More natural and fluid spoken interactions compared to baseline approaches

<img src="https://pub-ad90b2169561455ea151c5176b67b638.r2.dev/2025/07/20250707172902461.png" alt="Parrot" width="500"/>

## Installation

```bash
git clone https://github.com/Chaos96/NTPP.git
cd parrot
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1. Prepare audio data for pre-training and fine-tuning
2. Pre-train: `python pretrain.py --input_data path/to/single_channel_data`
3. Fine-tune: `python finetune.py --input_data path/to/double_channel_data`
4. Inference: `python inference.py --input_audio path/to/input.wav`
