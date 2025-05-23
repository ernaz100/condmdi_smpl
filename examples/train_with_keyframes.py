#!/usr/bin/env python3
"""
Example script for training STMC with keyframe conditioning.

This script demonstrates different keyframe conditioning strategies and 
provides a simple way to experiment with various settings.

Usage:
    python examples/train_with_keyframes.py --strategy standard
    python examples/train_with_keyframes.py --strategy inpainting --keyframes 10
    python examples/train_with_keyframes.py --strategy sparse --zero-loss
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to sys.path to import from stmc
sys.path.append(str(Path(__file__).parent.parent))


def run_training(strategy, **kwargs):
    """
    Run training with specified keyframe conditioning strategy.
    
    Args:
        strategy (str): Training strategy ('standard', 'inpainting', 'sparse')
        **kwargs: Additional parameters to override
    """
    
    # Base configuration
    config_overrides = []
    
    if strategy == "standard":
        # Standard keyframe conditioning for general motion completion
        config_overrides.extend([
            "diffusion.keyframe_conditioned=True",
            "diffusion.keyframe_selection_scheme=random_frames",
            "diffusion.zero_keyframe_loss=False",
            f"diffusion.n_keyframes={kwargs.get('keyframes', 5)}",
            f"diffusion.keyframe_mask_prob={kwargs.get('dropout', 0.1)}",
        ])
        
    elif strategy == "inpainting":
        # Focused inpainting training - loss only on non-keyframe regions
        config_overrides.extend([
            "diffusion.keyframe_conditioned=True",
            "diffusion.keyframe_selection_scheme=random_frames",
            "diffusion.zero_keyframe_loss=True",
            f"diffusion.n_keyframes={kwargs.get('keyframes', 8)}",
            f"diffusion.keyframe_mask_prob={kwargs.get('dropout', 0.15)}",
        ])
        
    elif strategy == "sparse":
        # Sparse keyframes for benchmarking
        config_overrides.extend([
            "diffusion.keyframe_conditioned=True",
            "diffusion.keyframe_selection_scheme=benchmark_sparse",
            f"diffusion.zero_keyframe_loss={kwargs.get('zero_loss', False)}",
            f"diffusion.trans_length={kwargs.get('trans_length', 10)}",
            f"diffusion.keyframe_mask_prob={kwargs.get('dropout', 0.1)}",
        ])
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Add any additional overrides
    if 'run_dir' in kwargs:
        config_overrides.append(f"run_dir={kwargs['run_dir']}")
    
    if 'batch_size' in kwargs:
        config_overrides.append(f"dataloader.batch_size={kwargs['batch_size']}")
        
    if 'max_epochs' in kwargs:
        config_overrides.append(f"trainer.max_epochs={kwargs['max_epochs']}")
    
    # Construct command
    cmd = [
        "python", "train.py",
        "--config-name=train_keyframe"
    ] + config_overrides
    
    print(f"Running training with strategy '{strategy}'")
    print(f"Command: {' '.join(cmd)}")
    print(f"Configuration overrides:")
    for override in config_overrides:
        print(f"  - {override}")
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train STMC with keyframe conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard keyframe conditioning
  python examples/train_with_keyframes.py --strategy standard
  
  # Inpainting-focused training with more keyframes
  python examples/train_with_keyframes.py --strategy inpainting --keyframes 10
  
  # Sparse conditioning with zero loss on keyframes
  python examples/train_with_keyframes.py --strategy sparse --zero-loss
  
  # Custom training with specific settings
  python examples/train_with_keyframes.py --strategy standard \\
    --keyframes 8 --dropout 0.2 --batch-size 64 --max-epochs 100
        """
    )
    
    parser.add_argument(
        "--strategy", 
        choices=["standard", "inpainting", "sparse"],
        required=True,
        help="Keyframe conditioning strategy"
    )
    
    parser.add_argument(
        "--keyframes", 
        type=int, 
        default=None,
        help="Number of keyframes (default varies by strategy)"
    )
    
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=None,
        help="Keyframe dropout probability (default varies by strategy)"
    )
    
    parser.add_argument(
        "--zero-loss", 
        action="store_true",
        help="Zero out loss on keyframe regions (for sparse strategy)"
    )
    
    parser.add_argument(
        "--trans-length", 
        type=int, 
        default=10,
        help="Transition length for sparse strategy (default: 10)"
    )
    
    parser.add_argument(
        "--run-dir", 
        type=str, 
        default=None,
        help="Custom run directory"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size override"
    )
    
    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=None,
        help="Maximum number of epochs"
    )
    
    args = parser.parse_args()
    
    # Prepare kwargs
    kwargs = {}
    if args.keyframes is not None:
        kwargs['keyframes'] = args.keyframes
    if args.dropout is not None:
        kwargs['dropout'] = args.dropout
    if args.zero_loss:
        kwargs['zero_loss'] = True
    if args.trans_length != 10:
        kwargs['trans_length'] = args.trans_length
    if args.run_dir:
        kwargs['run_dir'] = args.run_dir
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if args.max_epochs:
        kwargs['max_epochs'] = args.max_epochs
    
    # Run training
    run_training(args.strategy, **kwargs)


if __name__ == "__main__":
    main() 