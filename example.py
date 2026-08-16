# -*- coding: utf-8 -*-
"""Standalone example: caption an image with the local BLIP model.

Usage:
    python example.py "C:\\path\\to\\image.jpg"
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def main():
    if len(sys.argv) < 2:
        print("usage: python example.py <image_path>")
        return 1
    image_path = sys.argv[1]

    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    print(f"[caption] {caption}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
