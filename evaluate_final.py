import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import json
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

def generate_latex(model, processor, image, prompt, example_image=None, example_latex=None):
    messages = []
    if example_image is not None and example_latex is not None:
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Example: convert this formula to LaTeX"}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": example_latex}]})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images = [image]
    if example_image is not None:
        images.insert(0, example_image)
    inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "Assistant:" in generated_text:
        latex = generated_text.split("Assistant:")[-1].strip()
    else:
        latex = generated_text.strip()
    return latex.replace("\\boxed{", "").replace("}", "")

def normalize(s):
    return re.sub(r'\s+', '', s).lower()

def exact_match(ref, pred):
    return 1 if normalize(ref) == normalize(pred) else 0

def bleu_score(ref, pred):
    ref_t = list(normalize(ref))
    pred_t = list(normalize(pred))
    if not ref_t or not pred_t:
        return 0.0
    return sentence_bleu([ref_t], pred_t, weights=(0.25,0.25,0.25,0.25), smoothing_function=SmoothingFunction().method4)

test_dataset = load_dataset("linxy/LaTeX_OCR", "small", split="test")
train_dataset = load_dataset("linxy/LaTeX_OCR", "small", split="train")
example_image = train_dataset[0]["image"].convert("RGB")
example_latex = train_dataset[0]["text"]
prompt = "Convert this handwritten formula to LaTeX code. Output only the LaTeX."

# базовые модели (zero-shot и one-shot используют одну и ту же исходную модель)
print("Загружаем исходную модель SmolVLM...")
base_model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", torch_dtype=torch.float16).to(device)
base_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
base_model.eval()

# обученная на одном датасете
print("Загружаем модель, обученную на LaTeX_OCR...")
model_one = AutoModelForImageTextToText.from_pretrained("./model_latex_only/final", torch_dtype=torch.float16).to(device)
processor_one = AutoProcessor.from_pretrained("./model_latex_only/final")
model_one.eval()

# обученная на двух датасетах
# print("Загружаем модель, обученную на LaTeX_OCR + MathWriting...")
# model_two = AutoModelForImageTextToText.from_pretrained("./model_both/final", torch_dtype=torch.float16).to(device)
# processor_two = AutoProcessor.from_pretrained("./model_both/final")
# model_two.eval()

def evaluate_model(model, processor, name, is_baseline=False, use_one_shot=False):
    print(f"\n--- Оценка: {name} ---")
    predictions = []
    references = []
    for item in tqdm(test_dataset):
        image = item["image"].convert("RGB")
        correct = item["text"]
        if is_baseline and use_one_shot:
            pred = generate_latex(model, processor, image, prompt, example_image, example_latex)
        else:
            pred = generate_latex(model, processor, image, prompt)
        predictions.append(pred)
        references.append(correct)
    
    em = sum(exact_match(r, p) for r, p in zip(references, predictions)) / len(references)
    bleu = sum(bleu_score(r, p) for r, p in zip(references, predictions)) / len(references)
    print(f"  Exact Match: {em:.2%}")
    print(f"  BLEU: {bleu:.4f}")
    return {"em": em, "bleu": bleu}

results = {}
results["zero_shot"] = evaluate_model(base_model, base_processor, "Zero-shot (базовая)", is_baseline=True, use_one_shot=False)
results["one_shot"] = evaluate_model(base_model, base_processor, "One-shot (базовая)", is_baseline=True, use_one_shot=True)
results["sft_one"] = evaluate_model(model_one, processor_one, "SFT на LaTeX_OCR")
# results["sft_two"] = evaluate_model(model_two, processor_two, "SFT на LaTeX_OCR + MathWriting")

# with open("full_results.json", "w") as f:
#     json.dump(results, f, indent=2)