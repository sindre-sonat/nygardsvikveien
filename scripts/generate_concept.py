"""
Generate architectural concept images using local Stable Diffusion XL.

Uses RTX 4070 (8GB VRAM) with float16 precision.

Usage:
    uv run python scripts/generate_concept.py --prompt "description" --output output/concepts/name.png
    uv run python scripts/generate_concept.py --preset terrain    # Pre-defined concept prompts
    uv run python scripts/generate_concept.py --preset volume
    uv run python scripts/generate_concept.py --preset all
"""

import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


OUTPUT_DIR = Path(__file__).parent.parent / "output" / "concepts"

# Pre-defined prompts for the Nygårdsvikveien 38 project
PRESETS = {
    "terrain_good": {
        "prompt": (
            "Architectural cross-section drawing of a classical European house built into a steep hillside, "
            "stone retaining walls broken into short separate segments with green planted slopes and bushes between them, "
            "the basement floor partially embedded in natural terrain, soft organic terrain lines following the natural slope, "
            "native plants and shrubs growing between wall segments, the house appears to grow out of the hillside naturally, "
            "professional architectural section drawing style, clean white background, high detail"
        ),
        "negative": "flat terrain, continuous wall, platform, fortress, massive base, ugly, blurry",
        "filename": "concept_terrain_integrated.png",
    },
    "terrain_bad": {
        "prompt": (
            "Architectural cross-section drawing of a classical European house placed on top of a steep hillside, "
            "tall continuous stone retaining walls forming a massive unbroken platform base, "
            "the building sitting on a constructed platform above the natural terrain, "
            "extensive flat plateaus cut into the hillside, fortress-like stone base, "
            "professional architectural section drawing style, clean white background"
        ),
        "negative": "plants between walls, broken walls, natural terrain, ugly, blurry",
        "filename": "concept_terrain_dominant.png",
    },
    "volume_recessed": {
        "prompt": (
            "Architectural facade elevation drawing of a classical style three-story hillside house, "
            "the top floor is set back 0.8 meters from the main facade creating shadow lines, "
            "the stone base blends with terrain, upper floors are lighter in color, "
            "the building reads as two interlocking volumes rather than one box, "
            "hip roof with overhanging eaves, classical window proportions, "
            "professional architectural elevation drawing, clean white background, high detail"
        ),
        "negative": "one flat surface, no setback, brutalist, ugly, blurry",
        "filename": "concept_volume_recessed.png",
    },
    "hillside_reference": {
        "prompt": (
            "Beautiful classical European villa built into a steep green hillside in Bergen Norway, "
            "natural stone retaining walls in short segments with lush green vegetation between them, "
            "the house emerges naturally from the terrain, partially embedded in the slope, "
            "white painted facade with dark hip roof, terraced garden with native plants, "
            "overcast Nordic light, professional architectural photography, high quality"
        ),
        "negative": "flat terrain, massive walls, fortress, platform, ugly, blurry, cartoon",
        "filename": "concept_hillside_reference.png",
    },
    "before_after_terrain": {
        "prompt": (
            "Side by side architectural comparison drawing, left side shows a house sitting on tall continuous "
            "stone retaining walls like a fortress on a hillside with flat platforms, right side shows the same house "
            "with walls broken into segments with green planted slopes between them and terrain flowing naturally "
            "around the base, professional architectural illustration, labeled BEFORE and AFTER, white background"
        ),
        "negative": "photo, realistic, ugly, blurry",
        "filename": "concept_before_after.png",
    },
}


def load_pipeline():
    """Load SDXL pipeline optimized for 8GB VRAM."""
    print("Loading Stable Diffusion XL pipeline...")
    print("(First run will download ~6.5GB model weights)")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # Reduce VRAM usage

    print("Pipeline loaded successfully.")
    return pipe


def generate_image(pipe, prompt, negative_prompt="", output_path=None,
                   width=1024, height=768, steps=30, seed=None):
    """Generate a single image."""
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    print(f"Generating image ({width}x{height}, {steps} steps)...")
    print(f"Prompt: {prompt[:100]}...")
    start = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    )

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s")

    image = result.images[0]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Saved: {output_path}")

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate architectural concept images")
    parser.add_argument("--prompt", type=str, help="Custom prompt")
    parser.add_argument("--negative", type=str, default="ugly, blurry, low quality", help="Negative prompt")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()) + ["all"], help="Use a pre-defined prompt")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=768, help="Image height (default: 768)")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")

    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name}:")
            print(f"    {preset['prompt'][:80]}...")
            print(f"    -> {preset['filename']}")
            print()
        return

    if not args.prompt and not args.preset:
        parser.print_help()
        return

    pipe = load_pipeline()

    if args.preset:
        presets_to_run = PRESETS.items() if args.preset == "all" else [(args.preset, PRESETS[args.preset])]

        for name, preset in presets_to_run:
            print(f"\n{'='*60}")
            print(f"Generating preset: {name}")
            print(f"{'='*60}")
            output_path = str(OUTPUT_DIR / preset["filename"])
            generate_image(
                pipe,
                prompt=preset["prompt"],
                negative_prompt=preset.get("negative", ""),
                output_path=output_path,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
            )
    else:
        output_path = args.output or str(OUTPUT_DIR / "custom_concept.png")
        generate_image(
            pipe,
            prompt=args.prompt,
            negative_prompt=args.negative,
            output_path=output_path,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
