import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--model_path", type=str, help="Path to Model")
    parser.add_argument("--prompt", type=str, default="A <object> backpack", help="Prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of Images")
    parser.add_argument("--output_path", type=str, default="output", help="Path to output directory")

    args, unknown_args = parser.parse_known_args()

    model_id = args.model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

    prompt = args.prompt

    num_samples = args.num_samples

    all_images = []
    with autocast("cuda"):
        images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5)["sample"]
        all_images.extend(images)

    os.makedirs(args.output_path, exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(args.output_path, f"{i}.png"))