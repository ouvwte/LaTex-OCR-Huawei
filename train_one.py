import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
# for reproducibility
from utils import set_seed
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)

def format_dataset(example):
    image = example["image"].convert("RGB")
    latex = example["text"]
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this handwritten formula to LaTeX. Output only LaTeX."}]},
        {"role": "assistant", "content": [{"type": "text", "text": latex}]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = load_dataset("linxy/LaTeX_OCR", "small", split="train")
print(f"data size: {len(dataset)}")

dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)
print("data ready")

training_args = TrainingArguments(
    output_dir="./model_latex_only",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
)

trainer.train()
model.save_pretrained("./model_latex_only/final")
processor.save_pretrained("./model_latex_only/final")
print("Model saved into ./model_latex_only/final")