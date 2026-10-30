"""CER/WER on a JSONL split. Run once with --adapter and once without to see what tuning bought."""
import argparse
import json

import jiwer
import torch
from PIL import Image
from tqdm import tqdm

from common import MODEL_ID, load_model, load_processor, normalize, user_message


@torch.inference_mode()
def transcribe(model, proc, image, max_new_tokens=256):
    enc = proc.apply_chat_template(
        [user_message(image)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, enc["input_ids"].shape[1]:]
    return normalize(proc.batch_decode(new, skip_special_tokens=True)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit to score the untuned base model")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preds", default="preds.jsonl")
    args = ap.parse_args()

    proc = load_processor(args.model)
    model = load_model(args.model, adapter=args.adapter).eval()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    refs, hyps = [], []
    with open(args.preds, "w", encoding="utf-8") as out:
        for r in tqdm(rows):
            hyp = transcribe(model, proc, Image.open(r["image"]).convert("RGB"))
            refs.append(normalize(r["text"]))
            hyps.append(hyp)
            out.write(json.dumps({"image": r["image"], "ref": refs[-1], "hyp": hyp}, ensure_ascii=False) + "\n")

    print(f"samples={len(rows)}  CER={jiwer.cer(refs, hyps):.4f}  WER={jiwer.wer(refs, hyps):.4f}")
    print(f"per-sample predictions in {args.preds}")


if __name__ == "__main__":
    main()
