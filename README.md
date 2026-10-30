# Bangla handwriting fine-tune (Qwen3-VL-8B-Instruct, QLoRA)

Needs a CUDA GPU (about 16-24 GB VRAM for 4-bit). Run from `finetune/`.

```bash
pip install -r ../requirements.txt

# 0. BN-HTR -> labels.tsv (line crop <TAB> text <TAB> doc id)
python build_labels.py --dataset ../BN-HTR_Dataset --out labels.tsv

# 1. labels.tsv -> data/{train,val,test}.jsonl, split by document
python prepare_data.py --labels labels.tsv --image-root ../BN-HTR_Dataset --out data

# 2. train
python train.py --data data --out outputs/bangla-hw-lora

# 3. compare base vs tuned on the held-out documents
python evaluate.py --data data/test.jsonl --preds base_preds.jsonl
python evaluate.py --data data/test.jsonl --adapter outputs/bangla-hw-lora --preds tuned_preds.jsonl

# 4. use it
python infer.py line1.png line2.png
```

## BN-HTR notes

- Training samples are the line crops in `Segmentation_Images/Words/<doc>/<doc>_<page>/<doc>_<page>_<line>.jpg`.
  (`Segmentation_Images/Lines` holds whole pages with line boxes, not crops.)
- The per-document `.txt` is the typed source text and does not match the handwritten line breaks.
  Line labels are rebuilt from the per-word transcriptions in `<doc>.xlsx`.
- `build_labels.py` drops lines it cannot trust (duplicate word ids, word count != word-box count)
  and skips doc 84, whose sheet is a copy of doc 8's.
- Splits are by document, as the dataset has no writer ids; the same writer may appear in more than one split.
