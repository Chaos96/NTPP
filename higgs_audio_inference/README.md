# Higgs Audio Inference Package

这是 Higgs 双通道音频模型的推理打包版本，可作为独立子模块集成到其他项目中。

## 📦 打包内容

该包包含完整的 Higgs 音频模型推理所需的所有组件：

- **核心模型架构** (`boson_multimodal/model/higgs_audio/`)
  - 双通道音频生成模型
  - Transformer 编码器和解码器
  - 音频特征投影器
  - 延迟模式（delay pattern）支持
  - 多码本（codebook）音频生成

- **音频处理模块** (`boson_multimodal/audio_processing/`)
  - Higgs Audio Tokenizer（基于 DAC）
  - 语义编码器/解码器
  - 描述性音频编解码器（Descriptive Audio Codec）
  - 向量量化（Vector Quantization）

- **数据处理** (`boson_multimodal/data_collator/`, `boson_multimodal/dataset/`)
  - HiggsAudioSampleCollator（样本批处理）
  - ChatMLDatasetSample（对话数据结构）
  - 多通道音频 token 处理

- **推理脚本**
  - `infer_dual_channel.py` - 双通道音频生成推理脚本

## 📁 目录结构

```
higgs_audio_inference/
├── boson_multimodal/              # 核心库
│   ├── __init__.py
│   ├── constants.py               # Token 定义
│   ├── data_types.py              # ChatML 数据结构
│   ├── audio_processing/          # 音频 tokenizer + vocoder
│   │   ├── higgs_audio_tokenizer.py
│   │   ├── semantic_module.py
│   │   ├── descriptaudiocodec/    # DAC 编解码器
│   │   └── quantization/          # 向量量化
│   ├── data_collator/             # 数据批处理
│   │   └── higgs_audio_collator.py
│   ├── dataset/                   # 数据集工具
│   │   └── chatml_dataset.py
│   └── model/
│       └── higgs_audio/           # 核心模型
│           ├── modeling_higgs_audio.py      # 模型实现
│           ├── configuration_higgs_audio.py # 配置类
│           ├── audio_head.py                # 解码器投影
│           ├── utils.py                     # 工具函数
│           ├── common.py                    # 基类
│           ├── custom_modules.py            # 自定义层
│           └── cuda_graph_runner.py         # CUDA 优化
├── infer_dual_channel.py          # 推理脚本
├── requirements.txt               # 依赖列表
├── pyproject.toml                 # 项目配置（可选）
└── README.md                      # 本文档
```

## 🚀 快速开始

### 1. 打包模型

在当前目录（NTPP-higgs）下运行打包脚本：

```bash
chmod +x package_higgs.sh
./package_higgs.sh
```

这将创建 `higgs_audio_inference/` 文件夹，包含所有必需文件。

### 2. 集成到 NTPP 项目

将打包好的文件夹移动到你的 NTPP 项目中：

```bash
# 假设 NTPP 仓库位于 ../NTPP
cp -r higgs_audio_inference ../NTPP/

# 或者使用 git submodule（推荐）
cd ../NTPP
# 先将 higgs_audio_inference push 到独立仓库，然后：
# git submodule add <repo-url> higgs_audio_inference
```

### 3. 安装依赖

```bash
cd higgs_audio_inference
pip install -r requirements.txt
```

**核心依赖**：
- PyTorch >= 2.0
- Transformers >= 4.45.1, < 4.47.0
- descript-audio-codec
- librosa, torchaudio
- safetensors

### 4. 准备模型和数据

确保你有以下资源：

1. **模型检查点**（checkpoint）：
   ```
   outputs/dual_channel_sft_full/checkpoint-epoch2-step2999/
   ├── config.json
   ├── model.safetensors
   └── ...
   ```

2. **Tokenizer**：自动从 HuggingFace Hub 下载
   - 默认：`bosonai/higgs-audio-v2-tokenizer`

3. **测试数据**（可选）：tokenized 数据集
   ```
   dataset/tokenized_fisher/
   ├── val_manifest.jsonl
   └── tokens/
   ```

### 5. 运行推理

```bash
python infer_dual_channel.py \
    --checkpoint outputs/dual_channel_sft_full/checkpoint-epoch2-step2999 \
    --dataset-dir dataset/tokenized_fisher \
    --num-samples 5 \
    --output-dir outputs/inference_results \
    --device cuda \
    --max-frames 500
```

**参数说明**：
- `--checkpoint`: 模型检查点路径
- `--dataset-dir`: tokenized 数据集目录（包含 `val_manifest.jsonl`）
- `--num-samples`: 推理的样本数量
- `--output-dir`: 输出目录（生成的音频文件）
- `--device`: 设备选择（`cuda` 或 `cpu`）
- `--max-frames`: 最大帧数（用于加速测试）
- `--tokenizer`: Tokenizer 仓库（默认：`bosonai/higgs-audio-v2-tokenizer`）

## 💡 在代码中使用

你也可以将此包作为 Python 模块导入到你的 NTPP 项目中：

```python
# 在 NTPP 项目中
from higgs_audio_inference.boson_multimodal.model.higgs_audio import (
    HiggsAudioModel,
    HiggsAudioConfig
)
from higgs_audio_inference.boson_multimodal.audio_processing import (
    load_higgs_audio_tokenizer
)
from higgs_audio_inference.boson_multimodal.data_collator import (
    HiggsAudioSampleCollator
)

# 加载模型
config = HiggsAudioConfig.from_pretrained("path/to/checkpoint")
model = HiggsAudioModel(config).to("cuda")
model.load_state_dict(...)

# 加载 tokenizer
tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer")

# 创建 collator
collator = HiggsAudioSampleCollator(
    audio_in_token_id=128015,
    audio_out_token_id=128016,
    audio_stream_bos_id=1024,
    audio_stream_eos_id=1025,
    audio_num_codebooks=8,
    interleave_audio_channels=True,
    audio_token_frame_hz=50
)

# 推理流程
# ... (参考 infer_dual_channel.py 中的实现)
```

## 🔧 配置说明

### 模型配置

关键配置参数（在 `config.json` 中）：

```json
{
  "audio_num_codebooks": 8,          // 音频码本数量
  "audio_codebook_size": 1024,       // 每个码本的大小
  "audio_token_frame_hz": 50,        // 帧率（50 fps）
  "interleave_audio_channels": true, // 交错双通道
  "use_delay_pattern": false,        // 是否使用延迟模式
  "audio_dual_ffn_layers": [...]     // 双 FFN 层配置
}
```

### Token 规范

- **Audio-in token**: 128015 (`<|AUDIO|>`)
- **Audio-out token**: 128016 (`<|AUDIO_OUT|>`)
- **Audio stream BOS**: 1024
- **Audio stream EOS**: 1025
- **Pad token**: 0 或 128001
- **Text vocab size**: ~128000 (LLaMA-based)
- **Audio vocab size**: 1024 (每个码本)

## 🎯 推理输出

推理脚本会生成：

1. **音频文件**（WAV 格式）
   - `outputs/inference_results/sample_<id>_channel1_generated.wav`
   - 采样率：16000 Hz

2. **评估指标**（控制台输出）
   - RMSE（均方根误差）
   - MAE（平均绝对误差）
   - SNR（信噪比）
   - Correlation（相关系数）

3. **日志文件**
   - 生成过程的详细日志

## 🔍 故障排查

### 问题 1：找不到模块

**错误**：`ModuleNotFoundError: No module named 'boson_multimodal'`

**解决**：确保你在正确的目录下运行，或将 `higgs_audio_inference/` 添加到 Python 路径：

```python
import sys
sys.path.insert(0, '/path/to/higgs_audio_inference')
```

### 问题 2：CUDA 内存不足

**错误**：`RuntimeError: CUDA out of memory`

**解决**：
- 减少 `--max-frames` 参数
- 减少 `--num-samples`
- 使用 CPU：`--device cpu`

### 问题 3：Tokenizer 下载失败

**错误**：无法从 HuggingFace Hub 下载 tokenizer

**解决**：
- 检查网络连接
- 使用代理：`export HF_ENDPOINT=https://hf-mirror.com`
- 手动下载 tokenizer 并指定本地路径：`--tokenizer /path/to/local/tokenizer`

### 问题 4：导入路径错误

如果在 NTPP 项目中导入遇到问题，使用相对导入：

```python
# 假设 NTPP 结构：
# NTPP/
# ├── your_code.py
# └── higgs_audio_inference/

# 在 your_code.py 中：
from higgs_audio_inference.boson_multimodal.model.higgs_audio import HiggsAudioModel
```

## 📝 集成建议

### 方式 1：Git Submodule（推荐）

```bash
cd NTPP
git submodule add <higgs-inference-repo-url> higgs_audio_inference
git submodule update --init --recursive
```

**优点**：
- 独立版本管理
- 易于更新
- 不污染主仓库历史

### 方式 2：直接复制

```bash
cp -r higgs_audio_inference NTPP/
cd NTPP
git add higgs_audio_inference
git commit -m "Add Higgs audio inference module"
```

**优点**：
- 简单直接
- 无需额外的 submodule 管理

**缺点**：
- 更新需要手动复制
- 增加主仓库体积

### 方式 3：Python 包安装

在 `higgs_audio_inference/` 中运行：

```bash
pip install -e .
```

**优点**：
- 作为标准 Python 包使用
- 导入路径更清晰

## 📚 参考资源

- **原始训练文档**：`DUAL_CHANNEL_TRAINING_README.md`
- **模型架构**：`boson_multimodal/model/higgs_audio/modeling_higgs_audio.py`
- **推理示例**：`infer_dual_channel.py`

## 🐛 常见问题

**Q: 这个包可以单独作为 pip 包发布吗？**

A: 可以。已包含 `pyproject.toml`，你可以：
```bash
pip install build
python -m build
pip install dist/higgs_audio_inference-*.whl
```

**Q: 如何在 NTPP 中调用推理？**

A: 两种方式：
1. 命令行调用：`python higgs_audio_inference/infer_dual_channel.py ...`
2. Python 导入：参考"在代码中使用"部分

**Q: 模型大小多大？**

A:
- 代码：~3800 行核心代码 + 依赖
- 模型权重：取决于具体 checkpoint（通常几百 MB 到几 GB）

**Q: 支持哪些 PyTorch 版本？**

A: PyTorch >= 2.0，推荐 2.1+。CUDA 11.8+ 或 12.1+。

## 📄 许可证

（根据你的原项目许可证填写）

## 🤝 贡献

如需修改或扩展功能：
1. 修改 `higgs_audio_inference/` 中的代码
2. 测试确保兼容性
3. 更新本 README
4. 如使用 submodule，提交到独立仓库

---

**版本**: 1.0.0
**更新日期**: 2025-10-27
**维护者**: [Your Name]
