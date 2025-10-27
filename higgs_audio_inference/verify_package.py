#!/usr/bin/env python3
"""
Verification script for Higgs Audio Inference Package
Checks if all required modules and files are present.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, required=True):
    """Check if a file exists and return status."""
    exists = Path(filepath).exists()
    status = "✓" if exists else ("✗" if required else "⚠")
    req_str = "required" if required else "optional"
    print(f"  {status} {filepath} ({req_str})")
    return exists

def main():
    print("=" * 60)
    print("Higgs Audio Inference Package Verification")
    print("=" * 60)
    print()

    base_dir = Path(__file__).parent
    all_good = True

    # Check core files
    print("1. Checking core library files...")
    files_to_check = [
        ("boson_multimodal/__init__.py", True),
        ("boson_multimodal/constants.py", True),
        ("boson_multimodal/data_types.py", True),
    ]
    for filepath, required in files_to_check:
        if not check_file_exists(base_dir / filepath, required) and required:
            all_good = False
    print()

    # Check model files
    print("2. Checking model files...")
    model_files = [
        "boson_multimodal/model/__init__.py",
        "boson_multimodal/model/higgs_audio/__init__.py",
        "boson_multimodal/model/higgs_audio/modeling_higgs_audio.py",
        "boson_multimodal/model/higgs_audio/configuration_higgs_audio.py",
        "boson_multimodal/model/higgs_audio/audio_head.py",
        "boson_multimodal/model/higgs_audio/utils.py",
        "boson_multimodal/model/higgs_audio/common.py",
        "boson_multimodal/model/higgs_audio/custom_modules.py",
    ]
    for filepath in model_files:
        if not check_file_exists(base_dir / filepath, True):
            all_good = False
    print()

    # Check audio processing files
    print("3. Checking audio processing files...")
    audio_files = [
        "boson_multimodal/audio_processing/__init__.py",
        "boson_multimodal/audio_processing/higgs_audio_tokenizer.py",
        "boson_multimodal/audio_processing/semantic_module.py",
    ]
    for filepath in audio_files:
        if not check_file_exists(base_dir / filepath, True):
            all_good = False
    print()

    # Check data processing files
    print("4. Checking data processing files...")
    data_files = [
        "boson_multimodal/data_collator/__init__.py",
        "boson_multimodal/data_collator/higgs_audio_collator.py",
        "boson_multimodal/dataset/__init__.py",
        "boson_multimodal/dataset/chatml_dataset.py",
    ]
    for filepath in data_files:
        if not check_file_exists(base_dir / filepath, True):
            all_good = False
    print()

    # Check inference script
    print("5. Checking inference script...")
    check_file_exists(base_dir / "infer_dual_channel.py", True)
    print()

    # Check configuration files
    print("6. Checking configuration files...")
    check_file_exists(base_dir / "requirements.txt", True)
    check_file_exists(base_dir / "README.md", False)
    check_file_exists(base_dir / "pyproject.toml", False)
    print()

    # Test imports
    print("7. Testing Python imports...")
    sys.path.insert(0, str(base_dir))

    try:
        from boson_multimodal.model.higgs_audio import (
            HiggsAudioConfig,
            HiggsAudioModel
        )
        print("  ✓ HiggsAudioConfig and HiggsAudioModel import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import model classes: {e}")
        all_good = False

    try:
        from boson_multimodal.data_collator.higgs_audio_collator import (
            HiggsAudioSampleCollator
        )
        print("  ✓ HiggsAudioSampleCollator import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import collator: {e}")
        all_good = False

    try:
        from boson_multimodal.dataset.chatml_dataset import (
            ChatMLDatasetSample
        )
        print("  ✓ ChatMLDatasetSample import successful")
    except ImportError as e:
        print(f"  ✗ Failed to import dataset classes: {e}")
        all_good = False

    print()

    # Final result
    print("=" * 60)
    if all_good:
        print("✅ All checks passed! Package is ready to use.")
        print()
        print("Next steps:")
        print("  1. Copy this folder to your NTPP repository")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Run inference: python infer_dual_channel.py --help")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        print()
        print("Possible solutions:")
        print("  - Re-run the packaging script: ./package_higgs.sh")
        print("  - Check that all source files exist in the original directory")
        return 1
    print("=" * 60)

if __name__ == "__main__":
    sys.exit(main())
