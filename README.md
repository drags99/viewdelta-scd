# ViewDelta: Text-Conditioned Scene Change Detection

ViewDelta is a generalized framework for Scene Change Detection (SCD) that uses natural language prompts to define what changes are relevant. Unlike traditional change detection methods that implicitly learn what constitutes a "relevant" change from dataset labels, ViewDelta allows users to explicitly specify at runtime what types of changes they care about through text prompts.

## Overview

Given two images captured at different times and a text prompt describing the type of change to detect (e.g., "vehicle", "driveway", or "all changes"), ViewDelta produces a binary segmentation mask highlighting the relevant changes. The model is trained jointly on multiple datasets (CSeg, PSCD, SYSU-CD, VL-CMU-CD) and can:

- Detect user-specified changes via natural language prompts
- Handle unaligned image pairs with viewpoint variations
- Generalize across diverse domains (street-view, satellite, indoor/outdoor scenes)
- Detect all changes or specific semantic categories

For more details, see the paper: [ViewDelta: Scaling Scene Change Detection through Text-Conditioning](https://arxiv.org/abs/2412.07612)

## Installation

### Prerequisites

**Note:** ViewDelta has only been tested on Linux with the following specific versions:

- Python 3.10
- CUDA 12.1 (for GPU acceleration)
- NVIDIA GPU (tested on RTX 4090, L40S, and A100 - other GPUs may work)
- [Pixi package manager](https://pixi.sh/latest/)

### Install Pixi

First, install the Pixi package manager:

```bash
# On Linux
curl -fsSL https://pixi.sh/install.sh | bash
```

For more installation options, visit: https://pixi.sh/latest/installation/

### Install ViewDelta Dependencies

Once Pixi is installed, clone the repository and install dependencies:

```bash
pixi install
```

This will automatically set up the environment with all required dependencies including PyTorch, transformers, and other libraries.

## Running the Model

### Install ViewDelta Model Weights
```bash
wget https://huggingface.co/hoskerelab/ViewDelta/resolve/main/viewdelta_checkpoint.pth
```

### Basic Usage

The repository includes an [inference.py](inference.py) script for running the model on image pairs. Here's how to use it:

1. **Prepare your images**: Place two images you want to compare in the repository directory.

2. **Download a pre-trained checkpoint**: You'll need a model checkpoint file (e.g., `model.pth`).

3. **Edit the inference script**: Modify [inference.py](inference.py) to specify your images and text prompt:

```python
image_a_list = ["before_image.jpg"]
image_b_list = ["after_image.jpg"]
text_list = ["vehicle"]  # or "all" for all changes, or specific objects like "building", "tree", etc.

# Path to your checkpoint
PATH_TO_CHECKPOINT = "path/to/checkpoint.pth"
```

4. **Run inference**:

```bash
pixi run python inference.py
```

### Output

The script generates several outputs:
- `{image_name}_mask_{text}.png`: The binary segmentation mask
- `{image_name}_image_a_overlay.png`: First image with changes highlighted

### Text Prompt Examples

ViewDelta supports various types of text prompts:

- **Detect all changes**: `"What are the differences?"`, `"Find any differences"`
- **Specific objects**: `"vehicle"`, `"building"`, `"tree"`, `"person"`
- **Multiple objects**: `"vehicle, sign, barrier"`, `"cars and pedestrians"`
- **Natural language**: `"Has any construction equipment been added?"`, `"What buildings have changed?"`

### Model Configuration

The model uses:
- **Text embeddings**: SigLip (superior vision-language alignment)
- **Image embeddings**: DINOv2 (frozen pretrained features)
- **Architecture**: Vision Transformer (ViT) with 12 layers
- **Input resolution**: Images are automatically resized to 256×256

## Citation

```bibtex
@inproceedings{Varghese2024ViewDeltaSS,
  title={ViewDelta: Scaling Scene Change Detection through Text-Conditioning},
  author={Subin Varghese and Joshua Gao and Vedhus Hoskere},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:280642249}
}
```
