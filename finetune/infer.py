"""Transcribe one or more line-crop images with the tuned adapter."""
import argparse

from PIL import Image

from common import MODEL_ID, load_model, load_processor
from evaluate import transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--adapter", default="outputs/bangla-hw-lora")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    proc = load_processor(args.model)
    model = load_model(args.model, adapter=args.adapter).eval()
    for path in args.images:
        print(f"{path}\t{transcribe(model, proc, Image.open(path).convert('RGB'))}")


if __name__ == "__main__":
    main()
