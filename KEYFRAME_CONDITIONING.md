# Keyframe Conditioning in STMC

This document describes the keyframe conditioning feature added to STMC, inspired by the approach used in diffusion-motion-inbetweening.

## Overview

Keyframe conditioning allows the model to be trained and used for motion inpainting and motion editing tasks by providing sparse keyframe observations as conditioning information. During training, random keyframes are sampled and provided to the model, while the remaining frames are treated as targets to be generated.

## Key Features

1. **Flexible Keyframe Selection**: Multiple schemes for selecting keyframes during training and inference
2. **Robust Training**: Keyframe dropout mechanism to improve model robustness
3. **Configurable Loss Computation**: Option to zero out loss on keyframe regions
4. **Integration with STMC**: Seamless integration with existing STMC architecture

## Configuration

### Basic Setup

To enable keyframe conditioning, use the `mdm_smpl_keyframe` configuration:

```yaml
# configs/train_keyframe.yaml
defaults:
  - text_encoder: clip
  - motion_loader: smplrifke
  - mdm_smpl_keyframe  # Use keyframe-enabled configuration
  - trainer
  - defaults
  - _self_
```

### Keyframe Conditioning Parameters

The following parameters control keyframe conditioning behavior:

```yaml
diffusion:
  # Core keyframe conditioning settings
  keyframe_conditioned: True
  keyframe_selection_scheme: "random_frames"
  keyframe_mask_prob: 0.1
  zero_keyframe_loss: False
  n_keyframes: 5
  
  denoiser:
    keyframe_conditioned: True  # Must match diffusion setting
```

#### Parameter Details

- **`keyframe_conditioned`**: Enable/disable keyframe conditioning
- **`keyframe_selection_scheme`**: Strategy for selecting keyframes:
  - `"random_frames"`: Random sparse keyframes (recommended for training)
  - `"benchmark_sparse"`: Regular sparse keyframes every N frames
  - `"benchmark_clip"`: Beginning and end frames (for inbetweening)
  - `"pelvis_only"`: Sparse keyframes for root conditioning
  - `"gmd_keyframes"`: Fixed number of random keyframes
  - `"uncond"`: No keyframes (unconditional generation)
- **`keyframe_mask_prob`**: Probability of dropping keyframes during training (0.0-1.0)
- **`zero_keyframe_loss`**: If True, compute loss only on non-keyframe regions
- **`n_keyframes`**: Number of keyframes for certain selection schemes

## Usage

### Training with Keyframe Conditioning

```bash
# Train with keyframe conditioning enabled
python train.py --config-name=train_keyframe

# Train with custom keyframe settings
python train.py --config-name=train_keyframe \
  diffusion.keyframe_selection_scheme=benchmark_sparse \
  diffusion.n_keyframes=10 \
  diffusion.zero_keyframe_loss=True
```

### Model Architecture Changes

The keyframe conditioning feature includes:

1. **Enhanced Diffusion Model** (`GaussianDiffusion`):
   - Keyframe mask generation during training
   - Keyframe-aware loss computation
   - Integration of clean keyframe data into noisy input

2. **Enhanced Transformer Denoiser** (`TransformerDenoiser`):
   - Keyframe token embeddings to distinguish keyframes from non-keyframes
   - Optional keyframe data embeddings for clean motion incorporation
   - Attention mechanism that can leverage keyframe information

## Implementation Details

### Keyframe Mask Generation

During training, keyframes are selected using the `get_keyframes_mask` function:

```python
keyframe_mask = get_keyframes_mask(
    data=batch["x"],
    lengths=batch["length"],
    edit_mode=self.keyframe_selection_scheme,
    n_keyframes=self.n_keyframes,
    device=x.device
)
```

### Keyframe Conditioning in Forward Pass

1. **Input Conditioning**: Clean motion data is injected at keyframe positions in the noisy input
2. **Attention Conditioning**: Keyframe embeddings inform the transformer about which frames are observed
3. **Loss Masking**: Loss computation can be restricted to non-keyframe regions

### Training Process

```python
# Generate keyframe mask
keyframe_mask = get_keyframes_mask(data, lengths, edit_mode)

# Apply keyframe dropout for robustness
if keyframe_mask_prob > 0:
    keyframe_mask = get_random_keyframe_dropout_mask(keyframe_mask, keyframe_mask_prob)

# Replace keyframe regions in noisy input with clean data
xt = xt * (~keyframe_mask_expanded) + x * keyframe_mask_expanded

# Compute loss with keyframe-aware masking
loss_mask = create_keyframe_loss_mask(keyframe_mask, motion_mask, zero_keyframe_loss)
```

## Training Strategies

### 1. Standard Keyframe Conditioning
```yaml
keyframe_conditioned: True
keyframe_selection_scheme: "random_frames"
zero_keyframe_loss: False
n_keyframes: 5
```
- Trains on both keyframe and non-keyframe regions
- Good for general motion completion tasks

### 2. Inpainting-Focused Training
```yaml
keyframe_conditioned: True
keyframe_selection_scheme: "random_frames"
zero_keyframe_loss: True
n_keyframes: 8
```
- Loss computed only on non-keyframe regions
- Better for pure inpainting tasks

### 3. Sparse Conditioning
```yaml
keyframe_conditioned: True
keyframe_selection_scheme: "benchmark_sparse"
trans_length: 10
zero_keyframe_loss: False
```
- Regular sparse keyframes for evaluation consistency
- Good for benchmarking against other methods

## Monitoring Training

The keyframe conditioning feature adds several metrics to monitor training:

- **`keyframe_ratio`**: Average fraction of frames that are keyframes
- **`non_keyframe_loss`**: Loss computed only on non-keyframe regions (when `zero_keyframe_loss=True`)

## Integration with Existing Codebase

The keyframe conditioning feature is designed to be:

1. **Backward Compatible**: Can be disabled by setting `keyframe_conditioned=False`
2. **Modular**: Keyframe logic is separated into utility functions
3. **Configurable**: All parameters can be set via configuration files
4. **Well-Documented**: Comprehensive comments explain the implementation

## Advanced Usage

### Custom Keyframe Selection

You can extend the keyframe selection by modifying `src/utils/keyframe_masking.py`:

```python
def custom_keyframe_selection(data, lengths, **kwargs):
    # Your custom keyframe selection logic
    keyframe_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    # ... implement your logic ...
    return keyframe_mask
```

### Multi-Stage Training

Consider training with different keyframe strategies:

1. **Stage 1**: Dense keyframes for stable training
2. **Stage 2**: Sparse keyframes for challenging inpainting
3. **Stage 3**: Mixed strategies for robustness

## Troubleshooting

### Common Issues

1. **Memory Usage**: Keyframe conditioning adds minimal memory overhead
2. **Training Stability**: Use keyframe dropout (`keyframe_mask_prob > 0`) for better robustness
3. **Loss Convergence**: If using `zero_keyframe_loss=True`, monitor `non_keyframe_loss`

### Debugging

Enable detailed logging to monitor keyframe conditioning:

```python
logger.info("Keyframe conditioning enabled with following settings:")
logger.info(f"  - Selection scheme: {keyframe_selection_scheme}")
logger.info(f"  - Keyframe ratio: {keyframe_ratio:.3f}")
```

## Performance Considerations

- **Training Speed**: ~5-10% overhead due to keyframe processing
- **Memory Usage**: Minimal additional memory required
- **Model Size**: Small increase due to keyframe embeddings

## Future Extensions

Potential areas for improvement:

1. **Joint-Specific Keyframes**: Condition on specific body parts
2. **Temporal Keyframes**: More sophisticated temporal keyframe patterns
3. **Adaptive Keyframes**: Dynamic keyframe selection based on motion complexity
4. **Multi-Modal Keyframes**: Integration with other conditioning modalities 