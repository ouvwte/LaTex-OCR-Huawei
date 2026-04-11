import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import nltk
nltk.download('punkt', quiet=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using: {device}")

# Загружаем обученную модель
model_path = "./model_latex_only/final"  # путь к сохранённой модели
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
model.eval()

def generate_latex(image, prompt):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "Assistant:" in generated_text:
        latex = generated_text.split("Assistant:")[-1].strip()
    else:
        latex = generated_text.strip()
    return latex.replace("\\boxed{", "").replace("}", "")

def normalize(s): return re.sub(r'\s+', '', s).lower()
def exact_match(ref, pred): return 1 if normalize(ref) == normalize(pred) else 0
def bleu_score(ref, pred):
    ref_t = list(normalize(ref)); pred_t = list(normalize(pred))
    if not ref_t or not pred_t: return 0.0
    return sentence_bleu([ref_t], pred_t, weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method4)

test_dataset = load_dataset("linxy/LaTeX_OCR", "small", split="train")
prompt = "Convert this handwritten formula to LaTeX code. Output only the LaTeX."

predictions, references = [], []
for item in test_dataset:
    image = item["image"].convert("RGB")
    ref = item["text"]
    pred = generate_latex(image, prompt)
    predictions.append(pred); references.append(ref)

em = sum(exact_match(r,p) for r,p in zip(references, predictions)) / len(references)
bleu = sum(bleu_score(r,p) for r,p in zip(references, predictions)) / len(references)
print(f"SFT (один датасет) - Exact Match: {em:.2%}, BLEU: {bleu:.4f}")