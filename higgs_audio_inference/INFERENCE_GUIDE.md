# Higgs Audio Inference Scripts Guide

This package provides two inference scripts for different audio processing scenarios.

## 📋 Overview

| Script | Input | Output | Use Case |
|--------|-------|--------|----------|
| `infer_single_channel.py` | Single-channel audio tokens | Generated audio | Audio reconstruction, enhancement, or processing |
| `infer_dual_channel.py` | Dual-channel audio tokens | Channel 1 generation conditioned on Channel 0 | Conversational AI, stereo audio, dialogue systems |

---

## 🎵 Single-Channel Inference

### Script: `infer_single_channel.py`

**Purpose**: Process single-channel audio inputs for tasks like reconstruction, enhancement, or audio-to-audio transformation.

### Input Format
- **Token shape**: `[codebooks=8, frames]`
- Single audio stream
- Supports extracting one channel from multi-channel datasets

### Usage

```bash
python infer_single_channel.py \
    --checkpoint path/to/checkpoint \
    --dataset-dir path/to/tokenized_dataset \
    --num-samples 5 \
    --output-dir outputs/single_channel_results \
    --device cuda \
    --max-frames 500 \
    --channel-index 0
```

### Key Parameters

- `--channel-index`: Which channel to extract from multi-channel data (0 or 1)
  - Default: 0
  - Useful when working with dual-channel datasets but only processing one channel

### Output Files

For each sample (e.g., `sample_001`):
```
outputs/single_channel_results/
└── sample_001/
    ├── output_generated.wav      # Generated audio output
    └── input_groundtruth.wav     # Original input for comparison
```

### Workflow

1. Load single-channel tokens from dataset
2. Create ChatML sample with audio tokens as input
3. Model processes and generates output tokens
4. Decode tokens to audio waveform
5. Save and evaluate results

### Example Code

```python
from infer_single_channel import (
    load_model, load_dev_samples, prepare_input_batch,
    generate_audio, decode_to_audio
)

# Load model and samples
model = load_model("path/to/checkpoint", "cuda")
samples = load_dev_samples("path/to/dataset", num_samples=1, max_frames=500, channel_index=0)

# Process sample
sample_tokens = samples[0]['tokens']  # [8, frames]
sample = prepare_input_batch(sample_tokens, "cuda")

# Generate
output_tokens = generate_audio(model, collator, sample, "cuda")
audio = decode_to_audio(tokenizer, output_tokens)
```

---

## 🎧 Dual-Channel Inference

### Script: `infer_dual_channel.py`

**Purpose**: Generate Channel 1 audio conditioned on Channel 0 input, useful for conversational AI and dialogue systems.

### Input Format
- **Token shape**: `[channels=2, codebooks=8, frames]`
- Channel 0: Conditioning/input audio (e.g., speaker A)
- Channel 1: Target audio to generate (e.g., speaker B response)

### Usage

```bash
python infer_dual_channel.py \
    --checkpoint path/to/checkpoint \
    --dataset-dir path/to/tokenized_dataset \
    --num-samples 5 \
    --output-dir outputs/dual_channel_results \
    --device cuda \
    --max-frames 500
```

### Output Files

For each sample (e.g., `sample_001`):
```
outputs/dual_channel_results/
└── sample_001/
    ├── channel0_input.wav           # Input conditioning audio
    ├── channel1_generated.wav       # Generated output audio
    └── channel1_groundtruth.wav     # Ground truth for comparison
```

### Workflow

1. Load dual-channel tokens from dataset
2. Extract Channel 0 (conditioning) and Channel 1 (target)
3. Create ChatML sample with both channels
   - Channel 0 tokens marked as conditioning (labels = -100)
   - Channel 1 tokens used as target (supervised)
4. Model generates Channel 1 conditioned on Channel 0
5. Decode both channels and evaluate

### Example Code

```python
from infer_dual_channel import (
    load_model, load_dev_samples, prepare_input_batch,
    generate_channel1, decode_to_audio
)

# Load model and samples
model = load_model("path/to/checkpoint", "cuda")
samples = load_dev_samples("path/to/dataset", num_samples=1, max_frames=500)

# Process sample
sample_tokens = samples[0]['tokens']  # [2, 8, frames]
sample = prepare_input_batch(sample_tokens, "cuda")

# Generate channel 1 from channel 0
ch1_tokens = generate_channel1(model, collator, sample, "cuda")
ch1_audio = decode_to_audio(tokenizer, ch1_tokens)
```

---

## 🔧 Common Parameters

Both scripts share these parameters:

### Required Paths
- `--checkpoint`: Model checkpoint directory
  - Must contain `config.json` and `model.safetensors`
- `--dataset-dir`: Tokenized dataset directory
  - Must contain `val_manifest.jsonl`
  - Token files referenced in manifest

### Processing Options
- `--num-samples`: Number of validation samples to process
  - Default: 5
- `--max-frames`: Maximum audio frames (for speed control)
  - Default: 500
  - 50 Hz frame rate → 500 frames = 10 seconds

### Device Selection
- `--device`: Computation device
  - Choices: `cuda`, `cpu`
  - Auto-fallback to CPU if CUDA unavailable
  - CUDA enables bf16 automatic mixed precision

### Model Resources
- `--tokenizer`: HuggingFace tokenizer repo
  - Default: `bosonai/higgs-audio-v2-tokenizer`
  - Used for audio decoding (vocoder)

---

## 📊 Evaluation Metrics

Both scripts compute the same quality metrics:

### RMSE (Root Mean Squared Error)
- Measures overall prediction error
- Lower is better
- Range: [0, ∞)

### MAE (Mean Absolute Error)
- Average absolute difference
- Lower is better
- Range: [0, ∞)

### SNR (Signal-to-Noise Ratio)
- Ratio of signal power to noise power
- Higher is better
- Unit: dB (decibels)
- Formula: `10 * log10(signal_power / noise_power)`

### Correlation
- Pearson correlation coefficient
- Measures linear relationship
- Range: [-1, 1]
- 1 = perfect positive correlation

### Metrics Output

Results saved to `metrics.json`:
```json
{
  "per_sample": [
    {
      "sample_id": "sample_001",
      "rmse": 0.0234,
      "mae": 0.0189,
      "snr_db": 18.32,
      "correlation": 0.9567
    }
  ],
  "average": {
    "rmse": 0.0245,
    "mae": 0.0193,
    "snr_db": 17.89,
    "correlation": 0.9523
  }
}
```

---

## 🎯 Choosing the Right Script

### Use `infer_single_channel.py` when:
- ✅ Processing mono audio
- ✅ Audio enhancement tasks
- ✅ Audio reconstruction from tokens
- ✅ Working with single-speaker scenarios
- ✅ Extracting and processing one channel from stereo

### Use `infer_dual_channel.py` when:
- ✅ Conversational AI (dialogue generation)
- ✅ Turn-taking scenarios (speaker A → speaker B)
- ✅ Stereo audio processing
- ✅ Multi-speaker systems
- ✅ Generating responses conditioned on input

---

## 🔬 Technical Details

### Model Configuration

Both scripts use `HiggsAudioSampleCollator` with these settings:

#### Single-Channel
```python
collator = HiggsAudioSampleCollator(
    audio_in_token_id=128015,
    audio_out_token_id=128016,
    audio_num_codebooks=8,
    interleave_audio_channels=False,  # Single-channel mode
    audio_token_frame_hz=50,
)
```

#### Dual-Channel
```python
collator = HiggsAudioSampleCollator(
    audio_in_token_id=128015,
    audio_out_token_id=128016,
    audio_num_codebooks=8,
    interleave_audio_channels=True,  # Dual-channel mode
    audio_token_frame_hz=50,
)
```

### Token Specifications

- **Codebook size**: 1024 (tokens 0-1023)
- **Number of codebooks**: 8
- **Frame rate**: 50 Hz (50 frames per second)
- **Sample rate**: 16000 Hz (16 kHz)
- **Special tokens**:
  - Audio-in marker: 128015
  - Audio-out marker: 128016
  - Stream BOS: 1024
  - Stream EOS: 1025

### Data Format

#### ChatML Sample Structure
```python
ChatMLDatasetSample(
    input_ids=[128015, 128016],           # Text tokens
    label_ids=[-100, 128016],             # -100 = ignore
    audio_ids_concat=audio_tokens,        # [codebooks, total_frames]
    audio_ids_start=start_indices,        # Segment boundaries
    audio_ids_segment_channels=channels,  # Channel IDs
    audio_label_ids_concat=labels,        # Supervision labels
    ...
)
```

---

## 🐛 Troubleshooting

### Issue: "Expected token tensor with shape..."

**Problem**: Input token shape doesn't match expected format

**Solution**:
- **Single-channel**: Ensure tokens are `[8, frames]`
  - Use `--channel-index` to extract from multi-channel data
- **Dual-channel**: Ensure tokens are `[2, 8, frames]`

### Issue: "Model did not produce audio_logits"

**Problem**: Model forward pass failed

**Solution**:
- Check model checkpoint is compatible
- Verify CUDA memory if using GPU
- Try CPU mode: `--device cpu`

### Issue: "Predicted token sequence is empty"

**Problem**: All predictions were stream markers

**Solution**:
- Check collator configuration
- Verify `audio_stream_bos_id` and `audio_stream_eos_id` settings
- Ensure model is trained with same token scheme

### Issue: Token range warnings

**Problem**: Generated tokens outside valid range

**Solution**:
- Tokens are automatically clamped to [0, 1023]
- Check training data token distribution
- Verify model convergence

---

## 📚 Additional Resources

- **Main README**: `README.md` - Full package documentation
- **Training reference**: `DUAL_CHANNEL_TRAINING_README.md`
- **Package verification**: Run `python verify_package.py`

---

## 💡 Tips

1. **Start small**: Test with `--num-samples 1` and `--max-frames 100` first
2. **Use CUDA**: CPU inference is 10-50x slower
3. **Monitor memory**: Reduce `--max-frames` if OOM errors occur
4. **Compare metrics**: Use ground truth comparison to evaluate quality
5. **Check outputs**: Listen to generated audio files to verify quality

---

**Last Updated**: 2025-10-27
**Package Version**: 1.0.0
