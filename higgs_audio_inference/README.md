# Higgs Audio Inference

Audio generation and processing inference package based on the Higgs audio model architecture.

## 📦 Package Contents

This package provides complete inference capabilities for Higgs audio models:

- **Core Model Architecture** (`boson_multimodal/model/higgs_audio/`)
  - Dual-channel audio generation model
  - Transformer encoder and decoder
  - Audio feature projector
  - Delay pattern support
  - Multi-codebook audio generation

- **Audio Processing** (`boson_multimodal/audio_processing/`)
  - Higgs Audio Tokenizer (DAC-based)
  - Semantic encoder/decoder
  - Descriptive Audio Codec (DAC)
  - Vector Quantization (VQ)

- **Data Processing** (`boson_multimodal/data_collator/`, `boson_multimodal/dataset/`)
  - HiggsAudioSampleCollator (batch processing)
  - ChatMLDatasetSample (dialogue data structures)
  - Multi-channel audio token handling

- **Inference Scripts**
  - `infer_single_channel.py` - Single-channel audio inference
  - `infer_dual_channel.py` - Dual-channel audio generation

## 📁 Directory Structure

```
higgs_audio_inference/
├── boson_multimodal/              # Core library
│   ├── __init__.py
│   ├── constants.py               # Token definitions
│   ├── data_types.py              # ChatML data structures
│   ├── audio_processing/          # Audio tokenizer + vocoder
│   │   ├── higgs_audio_tokenizer.py
│   │   ├── semantic_module.py
│   │   ├── descriptaudiocodec/    # DAC codec
│   │   └── quantization/          # Vector quantization
│   ├── data_collator/             # Data batch processing
│   │   └── higgs_audio_collator.py
│   ├── dataset/                   # Dataset utilities
│   │   └── chatml_dataset.py
│   └── model/
│       └── higgs_audio/           # Core model
│           ├── modeling_higgs_audio.py      # Model implementation
│           ├── configuration_higgs_audio.py # Configuration classes
│           ├── audio_head.py                # Decoder projector
│           ├── utils.py                     # Utility functions
│           ├── common.py                    # Base classes
│           ├── custom_modules.py            # Custom layers
│           └── cuda_graph_runner.py         # CUDA optimization
├── infer_single_channel.py        # Single-channel inference script
├── infer_dual_channel.py          # Dual-channel inference script
├── INFERENCE_GUIDE.md             # Detailed inference guide
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

**Core Dependencies**:
- PyTorch >= 2.0
- Transformers >= 4.45.1, < 4.47.0
- descript-audio-codec
- librosa, torchaudio
- safetensors

### 2. Prepare Resources

Ensure you have the following:

1. **Model Checkpoint**:
   ```
   path/to/checkpoint/
   ├── config.json
   ├── model.safetensors
   └── ...
   ```

2. **Tokenizer**: Auto-downloaded from HuggingFace Hub
   - Default: `bosonai/higgs-audio-v2-tokenizer`

3. **Test Data** (optional): Tokenized dataset
   ```
   dataset/tokenized_data/
   ├── val_manifest.jsonl
   └── tokens/
   ```

### 3. Run Inference

#### Single-Channel Inference

For single-channel audio processing:

```bash
python infer_single_channel.py \
    --checkpoint path/to/checkpoint \
    --dataset-dir path/to/dataset \
    --num-samples 5 \
    --output-dir outputs/results \
    --device cuda \
    --channel-index 0
```

#### Dual-Channel Inference

For dual-channel audio generation (conversational AI):

```bash
python infer_dual_channel.py \
    --checkpoint path/to/checkpoint \
    --dataset-dir path/to/dataset \
    --num-samples 5 \
    --output-dir outputs/results \
    --device cuda \
    --max-frames 500
```

**Key Parameters**:
- `--checkpoint`: Path to model checkpoint directory
- `--dataset-dir`: Path to tokenized dataset directory (containing `val_manifest.jsonl`)
- `--num-samples`: Number of validation samples to process
- `--output-dir`: Output directory for generated audio files
- `--device`: Device to use (`cuda` or `cpu`)
- `--max-frames`: Maximum audio frames to generate (for speed control)
- `--tokenizer`: Tokenizer repo (default: `bosonai/higgs-audio-v2-tokenizer`)
- `--channel-index`: *(Single-channel only)* Channel to extract (0 or 1)

## 💡 Using as a Python Module

Import and use in your Python code:

```python
from boson_multimodal.model.higgs_audio import (
    HiggsAudioModel,
    HiggsAudioConfig
)
from boson_multimodal.audio_processing import (
    load_higgs_audio_tokenizer
)
from boson_multimodal.data_collator import (
    HiggsAudioSampleCollator
)

# Load model
config = HiggsAudioConfig.from_pretrained("path/to/checkpoint")
model = HiggsAudioModel(config).to("cuda")

# Load tokenizer
tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer")

# Create collator
collator = HiggsAudioSampleCollator(
    audio_in_token_id=128015,
    audio_out_token_id=128016,
    audio_stream_bos_id=1024,
    audio_stream_eos_id=1025,
    audio_num_codebooks=8,
    interleave_audio_channels=True,
    audio_token_frame_hz=50
)

# Run inference (see inference scripts for details)
```

## 🔧 Configuration

### Model Configuration

Key parameters in `config.json`:

```json
{
  "audio_num_codebooks": 8,          // Number of audio codebooks
  "audio_codebook_size": 1024,       // Size of each codebook
  "audio_token_frame_hz": 50,        // Frame rate (50 fps)
  "interleave_audio_channels": true, // Interleave dual channels
  "use_delay_pattern": false,        // Whether to use delay pattern
  "audio_dual_ffn_layers": [...]     // Dual FFN layer configuration
}
```

### Token Specifications

- **Audio-in token**: 128015 (`<|AUDIO|>`)
- **Audio-out token**: 128016 (`<|AUDIO_OUT|>`)
- **Audio stream BOS**: 1024
- **Audio stream EOS**: 1025
- **Pad token**: 0 or 128001
- **Text vocab size**: ~128000 (LLaMA-based)
- **Audio vocab size**: 1024 (per codebook)

## 🎯 Inference Outputs

The inference scripts generate:

1. **Audio Files** (WAV format)
   - Sample rate: 16000 Hz
   - Single-channel: `output_generated.wav`, `input_groundtruth.wav`
   - Dual-channel: `channel0_input.wav`, `channel1_generated.wav`, `channel1_groundtruth.wav`

2. **Evaluation Metrics** (console + JSON)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - SNR (Signal-to-Noise Ratio)
   - Correlation coefficient

3. **Metrics JSON**
   - Per-sample metrics
   - Average metrics across all samples

## 📊 Choosing the Right Script

### Use `infer_single_channel.py` when:
- ✅ Processing mono audio
- ✅ Audio enhancement tasks
- ✅ Audio reconstruction from tokens
- ✅ Single-speaker scenarios
- ✅ Extracting one channel from stereo

### Use `infer_dual_channel.py` when:
- ✅ Conversational AI (dialogue generation)
- ✅ Turn-taking scenarios
- ✅ Stereo audio processing
- ✅ Multi-speaker systems
- ✅ Generating responses conditioned on input

## 🔍 Troubleshooting

### Issue: Module not found

**Error**: `ModuleNotFoundError: No module named 'boson_multimodal'`

**Solution**: Ensure you're in the correct directory or add to Python path:

```python
import sys
sys.path.insert(0, '/path/to/higgs_audio_inference')
```

### Issue: CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce `--max-frames` parameter
- Reduce `--num-samples`
- Use CPU mode: `--device cpu`

### Issue: Tokenizer download failed

**Error**: Cannot download tokenizer from HuggingFace Hub

**Solution**:
- Check network connection
- Use proxy: `export HF_ENDPOINT=https://hf-mirror.com`
- Download tokenizer manually and specify local path: `--tokenizer /path/to/local/tokenizer`

### Issue: Token shape mismatch

**Error**: "Expected token tensor with shape..."

**Solution**:
- **Single-channel**: Ensure tokens are `[8, frames]`, use `--channel-index` if needed
- **Dual-channel**: Ensure tokens are `[2, 8, frames]`

## 📚 Documentation

- **Main README**: This file - Package overview and quick start
- **Inference Guide**: `INFERENCE_GUIDE.md` - Detailed inference documentation
- **Training Reference**: `DUAL_CHANNEL_TRAINING_README.md` - Training documentation

## 🐛 Common Questions

**Q: Can this be published as a pip package?**

A: Yes. The package includes `pyproject.toml`. You can build and install:
```bash
pip install build
python -m build
pip install dist/higgs_audio_inference-*.whl
```

**Q: What's the model size?**

A:
- Code: ~3800 lines of core code + dependencies
- Model weights: Depends on checkpoint (typically hundreds of MB to a few GB)

**Q: Which PyTorch versions are supported?**

A: PyTorch >= 2.0, recommended 2.1+. CUDA 11.8+ or 12.1+.

**Q: How do I use this in my project?**

A: Two ways:
1. Command-line: `python higgs_audio_inference/infer_*.py ...`
2. Python import: See "Using as a Python Module" section above

## 💡 Tips

1. **Start small**: Test with `--num-samples 1` and `--max-frames 100` first
2. **Use CUDA**: CPU inference is 10-50x slower
3. **Monitor memory**: Reduce `--max-frames` if OOM errors occur
4. **Check outputs**: Listen to generated audio to verify quality
5. **Read the guide**: See `INFERENCE_GUIDE.md` for comprehensive documentation

## 📄 License

Please refer to the original project license.

## 🙏 Acknowledgments

This package is built upon the [Higgs Audio project](https://github.com/boson-ai/higgs-audio) by Boson AI. We are grateful for their excellent work on audio language models and for making their research publicly available.

The Higgs audio model architecture, tokenizer, and training methodology are based on their original implementation. Please cite their work if you use this package in your research.

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
