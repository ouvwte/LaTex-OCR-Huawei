import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import json
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)
# for reproducibility
from utils import set_seed
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()
print("model is ready success!")

def generate_latex(image, prompt, example_image=None, example_latex=None):
    """
    Генерирует LaTeX по картинке.
    Если переданы example_image и example_latex - делает one-shot.
    """
    messages = []
    
    if example_image is not None and example_latex is not None:
        messages.append({
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Example: convert this formula to LaTeX"}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example_latex}]
        })
    
    messages.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]
    })
    
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
    
    latex = latex.replace("\\boxed{", "").replace("}", "")
    return latex

def normalize(latex):
    return re.sub(r'\s+', '', latex).lower()

def exact_match(ref, pred):
    return 1 if normalize(ref) == normalize(pred) else 0

def bleu_score(ref, pred):
    ref_tokens = list(normalize(ref))
    pred_tokens = list(normalize(pred))
    if not ref_tokens or not pred_tokens:
        return 0.0
    smoothing = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], pred_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothing)

test_dataset = load_dataset("linxy/LaTeX_OCR", "full", split="test")
test_dataset = test_dataset.select(range(70))
# print(f"examples count: {len(test_dataset)}")
example_image = test_dataset[0]["image"].convert("RGB")
example_latex = test_dataset[0]["text"] 
# train_dataset = load_dataset("linxy/LaTeX_OCR", "full", split="train")

prompt = "Convert this handwritten formula to LaTeX code. Output only the LaTeX."

results = {
    "zero_shot": {"predictions": [], "references": []},
    "one_shot": {"predictions": [], "references": []}
}

print("\nZero-shot: ")
for i, item in enumerate(tqdm(test_dataset)):
    image = item["image"].convert("RGB")
    correct = item["text"]
    pred = generate_latex(image, prompt)  # без примера
    results["zero_shot"]["predictions"].append(pred)
    results["zero_shot"]["references"].append(correct)

print("\nOne-shot:")
for i, item in enumerate(tqdm(test_dataset)):
    image = item["image"].convert("RGB")
    correct = item["text"]
    pred = generate_latex(image, prompt, example_image, example_latex)  # с примером
    results["one_shot"]["predictions"].append(pred)
    results["one_shot"]["references"].append(correct)

print("\nResults:")
output = {}

for method in ["zero_shot", "one_shot"]:
    refs = results[method]["references"]
    preds = results[method]["predictions"]
    
    em = sum(exact_match(r, p) for r, p in zip(refs, preds)) / len(refs)
    bleu = sum(bleu_score(r, p) for r, p in zip(refs, preds)) / len(refs)
    
    output[method] = {"exact_match_accuracy": em, "avg_bleu": bleu}
    print(f"\n{method.replace('_', ' ').title()}:")
    print(f"  Exact Match: {em:.2%}")
    print(f"  BLEU: {bleu:.4f}")

# with open("results.json", "w") as f:
#     json.dump(output, f, indent=2)