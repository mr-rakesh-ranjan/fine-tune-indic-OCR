from PIL import Image
from common import load_model, load_processor
from evaluate import transcribe

proc = load_processor()
model = load_model(adapter="outputs/bangla-hw-lora").eval()
print(transcribe(model, proc, Image.open(r"C:\\Users\\RakeshRanjanKumar\\RakeshDevelopments\\POCs_under_mayank\\IndicOCR_service\\finetune\\KYC_form_bangla.pdf").convert("RGB")))
