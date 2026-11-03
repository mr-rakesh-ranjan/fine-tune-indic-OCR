---
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
language:
- bn
tags:
- base_model:adapter:Qwen/Qwen3-VL-8B-Instruct
- lora
- qlora
- transformers
- ocr
- handwriting-recognition
- bangla
---

# bangla-hw-lora

A LoRA adapter for [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
that transcribes **one line of Bangla handwriting** from an image crop into Unicode text.
It was trained with QLoRA on line crops from the BN-HTR dataset.

> **Status:** training is complete, but character and word error rates (CER and WER) on the test
> set have not been measured yet. Until they are, the only quality signal is the validation loss
> shown below.

## Model details

- **Developed by:** Rakesh Ranjan
- **Model type:** LoRA adapter (QLoRA) for a vision-language model
- **Language:** Bangla (`bn`)
- **Finetuned from:** Qwen/Qwen3-VL-8B-Instruct
- **License:** not set yet (see [License](#license))

## What it does and does not do

**Works on:** a single cropped line of Bangla handwriting, similar to the BN-HTR line crops
(median about 1870 × 146 px).

**Not suited to:**

- **Whole pages or multi-line images.** It was never trained on them, and the processor caps images
  at about 0.8 MP, so a full page (BN-HTR pages are about 6.5 MP) is shrunk well below the
  resolution the model was trained on. Split the page into lines first, using a line detector, then
  transcribe each crop.
- **Printed text, other scripts or other languages.** These were not in the training data. For them,
  use the base model with the adapter switched off.
- **Layout, tables or reading order.** The adapter returns plain text for one line.

## How to use

Run these from `finetune/` in this repo. The model loads in 4-bit, which needs a CUDA GPU with about
16 GB of VRAM.

```bash
python infer.py line1.png line2.png --adapter outputs/bangla-hw-lora
```

From Python:

```python
from PIL import Image
from common import load_model, load_processor
from evaluate import transcribe

proc = load_processor()
model = load_model(adapter="outputs/bangla-hw-lora").eval()
print(transcribe(model, proc, Image.open("line.png").convert("RGB")))
```

The prompt must match the one used in training, which `common.PROMPT` provides:
`Transcribe the Bangla handwriting exactly. Output only the text.`
Outputs are NFC-normalised and whitespace-collapsed by `common.normalize`.

To use the base model for other content while this adapter is loaded, wrap the call in
`with model.disable_adapter():`.

## Training data

These are line crops from [BN-HTR](https://data.mendeley.com/datasets/743k6dm543), with labels
rebuilt from its per-word transcriptions (see the repo `README.md` for how the labels were built and
which lines were dropped). The data is split by document:

| Split | Lines | Documents |
|---|---:|---:|
| train | 10,920 | 118 |
| val | 1,840 | 15 |
| test | 1,274 | 15 |

The test split holds documents 4, 8, 42, 46, 50, 75, 80, 82, 83, 88, 92, 115, 126, 134 and 135.
BN-HTR has no writer IDs, so the same writer may appear in more than one split, and test scores may
therefore be optimistic for unseen writers.

## Training procedure

| Setting | Value |
|---|---|
| Method | QLoRA: 4-bit NF4 base model with double quantisation and bf16 compute |
| LoRA | r = 32, alpha = 64, dropout = 0.05, no bias |
| Target modules | `q/k/v/o/gate/up/down_proj` in the language model only; the vision tower is frozen |
| Loss | Applied to the assistant's transcription tokens only; the prompt and image tokens are masked |
| Epochs / steps | 3 epochs, 2,049 steps |
| Batch | 2 per device × 8 gradient-accumulation steps = 16 |
| Optimiser schedule | Learning rate 1e-4, cosine decay, 3% warmup |
| Image size | 200,704 to 802,816 pixels per crop |
| Precision | bf16, with gradient checkpointing |

### Validation loss

| Step | Epoch | Eval loss |
|---:|---:|---:|
| 200 | 0.29 | 0.3229 |
| 400 | 0.59 | 0.2437 |
| 600 | 0.88 | 0.2178 |
| 800 | 1.17 | 0.2154 |
| 1000 | 1.46 | 0.2085 |
| **1200** | **1.76** | **0.1923** |
| 1400 | 2.05 | 0.2040 |
| 1600 | 2.34 | 0.2113 |
| 1800 | 2.64 | 0.2121 |
| 2000 | 2.93 | 0.2123 |
| 2049 | 3.00 | 0.2122 |

**The weights in this folder are from step 1200**, the step with the best validation loss. They
were restored by `load_best_model_at_end`, and `adapter_model.safetensors` is byte-identical to
`checkpoint-1200`. Validation loss rose after epoch 2 while training loss kept falling (about 0.09
by step 800), so the later steps were overfitting. A future run could stop at about 2 epochs.

## Evaluation

| Model | CER | WER |
|---|---:|---:|
| Base Qwen3-VL-8B-Instruct (4-bit) | pending | pending |
| + this adapter | pending | pending |

To reproduce:

```bash
python evaluate.py --data data/test.jsonl --preds base_preds.jsonl
python evaluate.py --data data/test.jsonl --adapter outputs/bangla-hw-lora --preds tuned_preds.jsonl
```

## Compute and software

- **Hardware:** one NVIDIA GeForce RTX 4060 Ti with 16 GB, running Windows 11.
- **Wall time:** about 14 hours, from 2 Nov 2026 11:04 to 3 Nov 2026 01:08, including the
  evaluation passes every 200 steps.
- **Software:** transformers 5.22.1, PEFT 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 and
  torch 2.15.0+cu132.

## License

No license has been set yet. The base model is Apache-2.0; check the BN-HTR dataset's terms before
redistributing this adapter.
