# -*- coding: utf-8 -*-
"""Local image caption via BLIP (transformers + torch; CUDA first, CPU fallback).

Usage:
    python caption.py <image_path>

Prints one JSON line:
    {"ok": true, "caption": "...", "device": "cuda", "load_s": 3.1, "infer_s": 0.45}
"""
import sys
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Model is bundled in ./model (standard transformers format).
# If missing, it is downloaded from HuggingFace into ./model on first run.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_ID = "Salesforce/blip-image-captioning-base"


def run_caption(image_path, device):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    t0 = time.time()
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        processor = BlipProcessor.from_pretrained(MODEL_DIR)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    else:
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    t1 = time.time()

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    t2 = time.time()
    return {
        "ok": True,
        "caption": caption,
        "device": device,
        "load_s": round(t1 - t0, 1),
        "infer_s": round(t2 - t1, 2),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "missing image path"}, ensure_ascii=False))
        return 1
    image_path = sys.argv[1]
    try:
        import torch
        candidates = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        last_error = None
        for device in candidates:
            try:
                result = run_caption(image_path, device)
                print(json.dumps(result, ensure_ascii=False))
                return 0
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc).lower()
                if device == "cuda" and ("kernel image" in message or "cuda" in message or "out of memory" in message):
                    continue  # GPU unusable -> fall back to CPU
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
                return 1
        print(json.dumps({"ok": False, "error": str(last_error)}, ensure_ascii=False))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
