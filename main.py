import torch
from diffusers import ZImagePipeline
from datetime import datetime
from pathlib import Path
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate images with Z-Image")
parser.add_argument(
    "--prompt", "-p",
    type=str,
    default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
    help="Text prompt for image generation"
)
parser.add_argument(
    "--prompt-file",
    type=str,
    default=None,
    help="Path to a file containing the prompt text (takes precedence over --prompt)"
)
parser.add_argument(
    "--width", "-W",
    type=int,
    default=1024,
    help="Width of the generated image (default: 1024)"
)
parser.add_argument(
    "--height", "-H",
    type=int,
    default=1024,
    help="Height of the generated image (default: 1024)"
)
args = parser.parse_args()

# If prompt-file is provided, read its contents and use as prompt
if args.prompt_file:
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
else:
    prompt = args.prompt

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
# pipe.to("cuda")  # Disabled when using CPU offloading

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
pipe.enable_sequential_cpu_offload()

# 2. Generate Image
image = pipe(
    prompt=prompt,  # Use the prompt (from file or argument)
    height=args.height,
    width=args.width,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# Save to images directory with timestamp filename
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(output_dir / f"{timestamp}.png")