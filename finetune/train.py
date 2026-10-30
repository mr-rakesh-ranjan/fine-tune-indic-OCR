"""QLoRA fine-tune of Qwen3-VL-8B-Instruct on Bangla handwriting line crops."""
import argparse
import json

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common import MODEL_ID, load_model, load_processor, user_message


class LineDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Masks everything except the assistant answer so loss is only on the transcription."""

    def __init__(self, proc):
        self.proc = proc

    def _encode(self, messages, generation_prompt):
        return self.proc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation_prompt,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    def __call__(self, batch):
        images = [Image.open(r["image"]).convert("RGB") for r in batch]
        prompts = [[user_message(im)] for im in images]
        fulls = [
            p + [{"role": "assistant", "content": [{"type": "text", "text": r["text"]}]}]
            for p, r in zip(prompts, batch)
        ]
        enc = self._encode(fulls, generation_prompt=False)
        labels = enc["input_ids"].clone()
        for i, p in enumerate(prompts):
            prompt_len = self._encode([p], generation_prompt=True)["input_ids"].shape[1]
            labels[i, :prompt_len] = -100
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/bangla-hw-lora")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--no-quant", action="store_true", help="bf16 LoRA instead of 4-bit (needs ~40GB)")
    args = ap.parse_args()

    proc = load_processor(args.model)
    model = load_model(args.model, quant4=not args.no_quant)
    if not args.no_quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Language-model projections only; the vision tower stays frozen.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        ),
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # already enabled on the model above
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=LineDataset(f"{args.data}/train.jsonl"),
        eval_dataset=LineDataset(f"{args.data}/val.jsonl"),
        data_collator=Collator(proc),
    )
    trainer.train()
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
