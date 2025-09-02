# PyTorch Stable Diffusion

A complete from-scratch implementation of Stable Diffusion using PyTorch. This project implements all the core components of the Stable Diffusion model, including the U-Net architecture, VAE encoder/decoder, CLIP text encoder, and DDPM sampling algorithm.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts
- **Image-to-Image Generation**: Transform existing images based on text prompts
- **Classifier-Free Guidance (CFG)**: Enhanced prompt following with configurable guidance scale
- **DDPM Sampling**: Denoising Diffusion Probabilistic Model sampling algorithm
- **Complete Architecture Implementation**: All components built from scratch
  - CLIP text encoder for prompt processing
  - VAE encoder/decoder for latent space operations
  - U-Net with ResNet blocks and attention mechanisms
  - Time embedding for diffusion timesteps

## Architecture Components

### Core Models
- **`pipeline.py`**: Main inference pipeline orchestrating all components
- **`diffusion.py`**: U-Net diffusion model with attention and ResNet blocks
- **`clip.py`**: CLIP text encoder for processing prompts
- **`encoder.py`**: VAE encoder for converting images to latent space
- **`decoder.py`**: VAE decoder for converting latents back to images
- **`attention.py`**: Self-attention and cross-attention mechanisms
- **`ddpm.py`**: DDPM sampler for the denoising process

### Utilities
- **`model_loader.py`**: Load pretrained Stable Diffusion weights
- **`model_converter.py`**: Convert standard checkpoint formats
- **`demo.ipynb`**: Interactive demo notebook
- **`add_noise.ipynb`**: Noise addition experiments

## Requirements

- Python 3.11.3
- PyTorch 2.0.1
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd pytorch-stable-diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
   - Stable Diffusion v1.5 checkpoint: `v1-5-pruned-emaonly.ckpt`
   - CLIP tokenizer files: `vocab.json` and `merges.txt`
   
   Place these in a `data/` directory structure as expected by the demo.

## Usage

### Basic Text-to-Image Generation

```python
import model_loader
import pipeline
from transformers import CLIPTokenizer
import torch

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and models
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = model_loader.preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", DEVICE)

# Generate image
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt="",  # Negative prompt
    do_cfg=True,
    cfg_scale=8,
    sampler_name="ddpm",
    n_inference_steps=50,
    seed=42,
    models=models,
    device=DEVICE,
    tokenizer=tokenizer,
)

# Convert to PIL Image
from PIL import Image
Image.fromarray(output_image).save("generated_image.png")
```

### Image-to-Image Generation

```python
from PIL import Image

# Load input image
input_image = Image.open("input.jpg")

# Generate with image conditioning
output_image = pipeline.generate(
    prompt="Transform this into a painting",
    input_image=input_image,
    strength=0.8,  # Higher = more change from original
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    device=DEVICE,
    tokenizer=tokenizer,
)
```

### Interactive Demo

For a complete interactive example, see `sd/demo.ipynb` which demonstrates both text-to-image and image-to-image generation with various parameters.

## Parameters

### Generation Parameters
- **`prompt`**: Text description of desired image
- **`uncond_prompt`**: Negative prompt (what to avoid)
- **`cfg_scale`**: Classifier-free guidance scale (1-14, higher = more prompt adherence)
- **`strength`**: For img2img, how much to change input (0-1, higher = more change)
- **`n_inference_steps`**: Number of denoising steps (more = higher quality, slower)
- **`seed`**: Random seed for reproducible results

### Model Configuration
- **Input Resolution**: 512x512 pixels
- **Latent Resolution**: 64x64 (8x downscaled)
- **Text Sequence Length**: 77 tokens
- **Sampling Algorithm**: DDPM (Denoising Diffusion Probabilistic Model)

## Model Architecture Details

### U-Net Diffusion Model
- ResNet blocks with time embedding
- Multi-head self-attention and cross-attention layers
- Skip connections between encoder and decoder
- Group normalization and SiLU activation

### VAE (Variational Autoencoder)
- Encoder: RGB image → 4-channel latent representation
- Decoder: 4-channel latent → RGB image
- 8x spatial compression ratio

### CLIP Text Encoder
- Transformer-based architecture
- 77-token sequence length
- Learned positional embeddings
- Output serves as cross-attention conditioning

## Implementation Notes

- **Memory Efficient**: Models can be moved between GPU/CPU as needed
- **Modular Design**: Each component is independently implemented and testable
- **Educational**: Code is written for clarity and understanding
- **Compatible**: Uses standard Stable Diffusion checkpoint format

## Performance

- **Generation Time**: ~1-2 minutes on GPU for 50 inference steps
- **Memory Usage**: ~4-6GB VRAM for 512x512 generation
- **Quality**: Comparable to original Stable Diffusion v1.5

## Educational Value

This implementation is designed for learning and understanding:
- How diffusion models work at a low level
- The role of each component in the generation pipeline
- Attention mechanisms in computer vision
- VAE latent space representations
- Text-to-image conditioning via cross-attention

## License

Please refer to the original Stable Diffusion license and terms of use when using pretrained weights.

## Acknowledgments

Based on the Stable Diffusion architecture by Stability AI, with reference implementations and the original paper "High-Resolution Image Synthesis with Latent Diffusion Models".
