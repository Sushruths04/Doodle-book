"""
Dataset preparation for LoRA training.

Prepares character images for DreamBooth-style fine-tuning.
"""

import os
from PIL import Image


def prepare_training_data(
    input_dir: str,
    output_dir: str = "./training_data",
    target_size: int = 512,
    num_augmentations: int = 5
):
    """
    Prepare training images for LoRA fine-tuning.
    
    Steps:
    1. Load original doodle images
    2. Resize to target size
    3. Create variations (flip, rotate, color shift)
    4. Save with consistent naming
    
    Args:
        input_dir: Directory with original doodle images
        output_dir: Output directory for prepared data
        target_size: Target image size (512x512 for FLUX)
        num_augmentations: Number of augmented versions per image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 5: Full implementation with augmentations
    raise NotImplementedError("Phase 5: Dataset preparation")


if __name__ == "__main__":
    print("Run this script to prepare training data for LoRA.")
