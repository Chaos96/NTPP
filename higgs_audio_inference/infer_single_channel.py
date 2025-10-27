"""Inference script for single-channel audio generation.

This script:
1. Loads the trained audio model
2. Takes single-channel audio tokens as input
3. Generates audio output tokens
4. Decodes to audio waveform for quality evaluation
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
from tqdm import tqdm

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-channel audio inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/dual_channel_sft_full/checkpoint-epoch2-step2999",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/tokenized_fisher",
        help="Path to tokenized dataset directory containing validation manifest"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of validation samples to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference_results",
        help="Output directory for generated audio files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of audio frames to generate (for speed control)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="HuggingFace tokenizer repo path for audio vocoder"
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index to use from multi-channel dataset (0 or 1)"
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> torch.device:
    """
    Resolve the device to use for inference.

    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        device_choice: Device string ('cuda' or 'cpu')

    Returns:
        torch.device instance
    """
    if device_choice.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_choice)
        print("⚠️  CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: Union[str, torch.device]) -> HiggsAudioModel:
    """
    Load trained model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory containing config.json and model.safetensors
        device: Device to load model on

    Returns:
        Loaded HiggsAudioModel in eval mode
    """
    print(f"\n🤖 Loading model from {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Load configuration
    config = HiggsAudioConfig.from_pretrained(checkpoint_path)

    # Initialize model
    model = HiggsAudioModel(config)

    # Load weights from safetensors
    weights_path = checkpoint_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")

    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model loaded with {num_params:,} parameters")

    return model


def load_dev_samples(
    dataset_dir: str,
    num_samples: int,
    max_frames: int,
    channel_index: int = 0
):
    """
    Load validation samples from dataset manifest.

    Args:
        dataset_dir: Dataset directory path
        num_samples: Number of samples to load
        max_frames: Maximum frame length to truncate
        channel_index: Channel index to extract from multi-channel data (0 or 1)

    Returns:
        List of sample dictionaries with single-channel tokens
    """
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "val_manifest.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")

    print(f"\n📂 Loading validation samples from {manifest_path}")
    print(f"   Extracting channel {channel_index}")

    samples = []
    with open(manifest_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            sample_meta = json.loads(line)

            # Load token file
            token_path = dataset_path / sample_meta['token_path']
            if not token_path.exists():
                print(f"   ⚠️  Missing token file for {sample_meta['id']}: {token_path}, skipping.")
                continue

            # Load tokens: expected shape [channels, codebooks=8, frames] or [codebooks=8, frames]
            tokens = torch.load(token_path, map_location="cpu")

            # Handle multi-channel data by extracting specified channel
            if tokens.ndim == 3:  # [channels, codebooks, frames]
                if tokens.shape[0] <= channel_index:
                    print(f"   ⚠️  Channel {channel_index} not found in {sample_meta['id']}, skipping.")
                    continue
                tokens = tokens[channel_index]  # Extract single channel: [codebooks=8, frames]
            elif tokens.ndim == 2:  # Already [codebooks, frames]
                pass  # Use as-is
            else:
                print(f"   ⚠️  Unexpected token shape {tokens.shape} for {sample_meta['id']}, skipping.")
                continue

            # Truncate if needed
            if tokens.shape[1] > max_frames:
                tokens = tokens[:, :max_frames]

            samples.append({
                'id': sample_meta['id'],
                'tokens': tokens,  # [codebooks=8, frames]
                'duration': tokens.shape[1] / 50,  # 50 Hz frame rate
                'original_path': sample_meta.get('original_path', 'unknown')
            })

    if not samples:
        raise RuntimeError("No samples could be loaded from the validation manifest.")

    print(f"   ✓ Loaded {len(samples)} samples")
    for i, s in enumerate(samples):
        print(f"      [{i}] {s['id']}: {s['tokens'].shape} ({s['duration']:.1f}s)")

    return samples


def prepare_input_batch(sample_tokens: torch.Tensor, device: Union[str, torch.device]):
    """
    Prepare input batch for model inference.

    Args:
        sample_tokens: Single-channel input tokens [codebooks=8, frames]
        device: Device to move tensors to

    Returns:
        ChatMLDatasetSample for collator processing
    """
    if sample_tokens.ndim != 2 or sample_tokens.shape[0] != 8:
        raise ValueError(
            f"Expected token tensor with shape [codebooks=8, frames], got {sample_tokens.shape}"
        )

    # Input audio tokens
    input_tokens = sample_tokens.to(dtype=torch.long, copy=False)

    # For single-channel: we use input as both conditioning and target
    # This is suitable for audio reconstruction or enhancement tasks
    audio_ids_concat = input_tokens  # [codebooks=8, frames]
    audio_ids_start = torch.tensor([0], dtype=torch.long)
    audio_ids_segment_channels = torch.tensor([0], dtype=torch.long)

    # Labels: supervise all frames
    audio_label_ids_concat = input_tokens.clone()

    # Text token IDs for audio input/output markers
    audio_in_token_id = 128015
    audio_out_token_id = 128016
    input_ids = torch.tensor([audio_in_token_id, audio_out_token_id], dtype=torch.long)
    label_ids = input_ids.clone()
    label_ids[0] = -100  # Ignore loss for input marker

    # Create ChatML sample
    sample = ChatMLDatasetSample(
        input_ids=input_ids,
        label_ids=label_ids,
        audio_ids_concat=audio_ids_concat,
        audio_ids_start=audio_ids_start,
        audio_waveforms_concat=torch.tensor([], dtype=torch.float32),
        audio_waveforms_start=torch.tensor([], dtype=torch.long),
        audio_sample_rate=torch.tensor([], dtype=torch.long),
        audio_speaker_indices=torch.tensor([], dtype=torch.long),
        audio_label_ids_concat=audio_label_ids_concat,
        audio_ids_segment_channels=audio_ids_segment_channels,
        reward=None,
    )

    return sample


def sanitize_audio_tokens(tokens: torch.Tensor, max_token_id: int = 1023) -> torch.Tensor:
    """
    Clamp audio tokens to valid codec vocabulary range.

    Args:
        tokens: Audio token tensor
        max_token_id: Maximum valid token ID (default 1023 for 1024-sized codebook)

    Returns:
        Clamped tokens
    """
    return tokens.clamp_(min=0, max=max_token_id)


def strip_stream_tokens(
    tokens: torch.Tensor,
    bos_id: int = 1024,
    eos_id: int = 1025,
) -> torch.Tensor:
    """
    Remove audio stream BOS/EOS markers inserted by collator.

    Args:
        tokens: Token tensor [codebooks, seq_len]
        bos_id: Beginning-of-stream token ID
        eos_id: End-of-stream token ID

    Returns:
        Filtered tokens without stream markers
    """
    if tokens.numel() == 0:
        return tokens
    # Check first codebook for stream markers
    mask = (tokens[0] != bos_id) & (tokens[0] != eos_id)
    return tokens[:, mask]


@torch.no_grad()
def generate_audio(
    model: HiggsAudioModel,
    collator: HiggsAudioSampleCollator,
    sample: ChatMLDatasetSample,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    Generate audio tokens from input using the model.

    Args:
        model: Trained HiggsAudioModel
        collator: Data collator for batch preparation
        sample: Input sample with audio tokens
        device: Inference device

    Returns:
        Generated audio tokens: [codebooks=8, frames]
    """
    # Collate into batch format
    batch = collator([sample])

    device_obj = torch.device(device)

    # Prepare batch dictionary for model forward pass
    batch_dict = {
        'input_ids': batch.input_ids.to(device_obj),
        'attention_mask': batch.attention_mask.to(device_obj),
        'audio_in_ids': batch.audio_in_ids.to(device_obj) if batch.audio_in_ids is not None else None,
        'audio_in_ids_start': batch.audio_in_ids_start.to(device_obj) if batch.audio_in_ids_start is not None else None,
        'audio_out_ids': batch.audio_out_ids.to(device_obj),
        'audio_out_ids_start': batch.audio_out_ids_start.to(device_obj),
        'audio_out_ids_start_group_loc': batch.audio_out_ids_start_group_loc.to(device_obj) if batch.audio_out_ids_start_group_loc is not None else None,
        'label_audio_ids': batch.label_audio_ids.to(device_obj) if batch.label_audio_ids is not None else None,
        'audio_in_position_ids': batch.audio_in_position_ids.to(device_obj) if batch.audio_in_position_ids is not None else None,
        'audio_in_channel_ids': batch.audio_in_channel_ids.to(device_obj) if batch.audio_in_channel_ids is not None else None,
        'audio_out_position_ids': batch.audio_out_position_ids.to(device_obj) if batch.audio_out_position_ids is not None else None,
        'audio_out_channel_ids': batch.audio_out_channel_ids.to(device_obj) if batch.audio_out_channel_ids is not None else None,
    }

    # Run forward pass with automatic mixed precision on CUDA
    autocast_enabled = device_obj.type == "cuda"
    autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if autocast_enabled else nullcontext()

    with autocast_context:
        outputs = model(**batch_dict)

    # Extract predictions from logits
    if outputs.audio_logits is None:
        raise ValueError("Model did not produce audio_logits output")

    logits = outputs.audio_logits  # [seq_len, num_codebooks, vocab_size]
    predicted = torch.argmax(logits, dim=-1).transpose(0, 1).contiguous()  # [num_codebooks, seq_len]

    # Remove stream markers
    predicted = strip_stream_tokens(predicted)
    if predicted.numel() == 0:
        raise ValueError("Predicted token sequence is empty after removing stream markers.")

    # Sanitize to valid range
    output_tokens = sanitize_audio_tokens(predicted.clone())

    return output_tokens


def decode_to_audio(tokenizer, tokens: torch.Tensor) -> np.ndarray:
    """
    Decode audio tokens to waveform using vocoder.

    Args:
        tokenizer: Higgs audio tokenizer with vocoder
        tokens: Audio tokens [codebooks=8, frames]

    Returns:
        Audio waveform as numpy array
    """
    # Add batch dimension
    tokens = tokens.to(dtype=torch.long, copy=False).unsqueeze(0)  # [1, 8, frames]

    # Decode through vocoder
    audio_tensor = tokenizer.decode(tokens)  # Returns [batch, channels, samples]

    # Extract single-channel audio
    if isinstance(audio_tensor, torch.Tensor):
        audio_numpy = audio_tensor[0, 0].detach().cpu().numpy()
    else:
        audio_numpy = audio_tensor[0, 0]

    return audio_numpy


def save_audio(audio: np.ndarray, path: Path, sample_rate: int):
    """
    Save audio waveform to WAV file.

    Args:
        audio: Audio waveform (float32, range approximately [-1, 1])
        path: Output file path
        sample_rate: Audio sample rate in Hz
    """
    # Convert to int16 format
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)


def compute_metrics(pred_audio: np.ndarray, gt_audio: np.ndarray) -> dict:
    """
    Compute audio quality metrics between prediction and ground truth.

    Args:
        pred_audio: Predicted audio waveform
        gt_audio: Ground truth audio waveform

    Returns:
        Dictionary with RMSE, MAE, SNR, and correlation metrics
    """
    # Ensure same length for comparison
    min_len = min(len(pred_audio), len(gt_audio))
    pred_audio = pred_audio[:min_len]
    gt_audio = gt_audio[:min_len]

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((pred_audio - gt_audio) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(pred_audio - gt_audio))

    # Signal-to-Noise Ratio
    signal_power = np.mean(gt_audio ** 2)
    noise_power = np.mean((pred_audio - gt_audio) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Pearson Correlation
    corr = np.corrcoef(pred_audio, gt_audio)[0, 1]

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'snr_db': float(snr),
        'correlation': float(corr),
    }


def main():
    """Main inference pipeline."""
    args = parse_args()

    device = resolve_device(args.device)

    print("=" * 80)
    print("🎵 Single-Channel Audio Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max frames: {args.max_frames}")
    print(f"Channel index: {args.channel_index}")
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model checkpoint
    model = load_model(args.checkpoint, device)

    # Load tokenizer/vocoder
    print(f"\n🔧 Loading tokenizer/vocoder: {args.tokenizer}")
    tokenizer = load_higgs_audio_tokenizer(args.tokenizer, device=str(device))
    sample_rate = tokenizer.sampling_rate
    print(f"   ✓ Sample rate: {sample_rate} Hz")

    # Create data collator
    print("\n🔧 Creating data collator")
    collator = HiggsAudioSampleCollator(
        whisper_processor=None,
        audio_in_token_id=128015,
        audio_out_token_id=128016,
        pad_token_id=0,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        encode_whisper_embed=False,
        return_audio_in_tokens=True,
        audio_num_codebooks=8,
        use_delay_pattern=False,
        interleave_audio_channels=False,  # Single-channel mode
        audio_token_frame_hz=50,
    )

    # Load validation samples
    samples = load_dev_samples(
        args.dataset_dir,
        args.num_samples,
        args.max_frames,
        args.channel_index
    )

    # Process each sample
    print("\n" + "=" * 80)
    print("🎯 Running Inference")
    print("=" * 80)

    all_metrics = []

    for i, sample_data in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing {sample_data['id']}")

        # Prepare input batch
        sample = prepare_input_batch(sample_data['tokens'], device)

        # Generate audio tokens
        print("   Generating audio tokens...")
        try:
            generated_tokens = generate_audio(model, collator, sample, device)
            print(f"   ✓ Generated tokens shape: {generated_tokens.shape}")
            generated_tokens = sanitize_audio_tokens(generated_tokens.clone())
        except Exception as e:
            print(f"   ✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Get ground truth tokens (use same input tokens for comparison)
        gt_tokens = sample_data['tokens'][:, :generated_tokens.shape[1]]

        # Decode to audio waveform
        print("   Decoding to audio waveform...")
        try:
            print(f"   Generated token range: min={generated_tokens.min().item()}, max={generated_tokens.max().item()}")
            print(f"   Ground truth token range: min={gt_tokens.min().item()}, max={gt_tokens.max().item()}")

            generated_audio = decode_to_audio(tokenizer, generated_tokens.cpu())
            gt_audio = decode_to_audio(tokenizer, gt_tokens.cpu())

            print(f"   ✓ Generated audio: {len(generated_audio)} samples ({len(generated_audio)/sample_rate:.2f}s)")
        except Exception as e:
            print(f"   ✗ Decoding failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save audio files
        sample_output_dir = output_dir / sample_data['id']
        sample_output_dir.mkdir(exist_ok=True)

        save_audio(generated_audio, sample_output_dir / "output_generated.wav", sample_rate)
        save_audio(gt_audio, sample_output_dir / "input_groundtruth.wav", sample_rate)

        print(f"   ✓ Saved audio to {sample_output_dir}")

        # Compute quality metrics
        metrics = compute_metrics(generated_audio, gt_audio)
        metrics['sample_id'] = sample_data['id']
        all_metrics.append(metrics)

        print(f"   📊 Metrics:")
        print(f"      RMSE: {metrics['rmse']:.4f}")
        print(f"      MAE: {metrics['mae']:.4f}")
        print(f"      SNR: {metrics['snr_db']:.2f} dB")
        print(f"      Correlation: {metrics['correlation']:.4f}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("📊 Summary Statistics")
    print("=" * 80)

    if all_metrics:
        avg_metrics = {
            'rmse': np.mean([m['rmse'] for m in all_metrics]),
            'mae': np.mean([m['mae'] for m in all_metrics]),
            'snr_db': np.mean([m['snr_db'] for m in all_metrics]),
            'correlation': np.mean([m['correlation'] for m in all_metrics]),
        }

        print(f"Average RMSE: {avg_metrics['rmse']:.4f}")
        print(f"Average MAE: {avg_metrics['mae']:.4f}")
        print(f"Average SNR: {avg_metrics['snr_db']:.2f} dB")
        print(f"Average Correlation: {avg_metrics['correlation']:.4f}")

        # Save results to JSON
        results_path = output_dir / "metrics.json"
        with open(results_path, 'w') as f:
            json.dump({
                'per_sample': all_metrics,
                'average': avg_metrics,
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_path}")

    print("\n" + "=" * 80)
    print("✅ Inference Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
