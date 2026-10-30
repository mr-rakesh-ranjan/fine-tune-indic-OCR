import unicodedata

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = "Transcribe the Bangla handwriting exactly. Output only the text."

# Visual tokens are 28x28 px patches after merging; this caps tokens per line crop.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


def normalize(text: str) -> str:
    """NFC plus whitespace cleanup, so labels and predictions compare consistently."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def load_processor(model_id: str = MODEL_ID):
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "right"
    return proc


def load_model(model_id: str = MODEL_ID, adapter: str | None = None, quant4: bool = True):
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if quant4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model


def user_message(image):
    return {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
    }
