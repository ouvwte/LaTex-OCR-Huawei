# Technical Report for Task 1: Fine-tuning Vision-Language Models for OCR Formulas

## 1. Experimental Setup

- **Model:** `HuggingFaceTB/SmolVLM-256M-Instruct`
- **Retraining Method:** LoRA (PEFT)
- **Training Hyperparameters:**
- Number of epochs: 3
- Batch size (effective): 4 (per_device=1, gradient_accumulation=4)
- Learning rate: 2e-4
- Optimizer: AdamW
- LoRA parameters: r=8, lora_alpha=16, target_modules=q_proj, v_proj, dropout=0.05
- Data type: FP16 (mixed precision)
- **Hardware:** NVIDIA GPU (CUDA) / CPU (if GPU is not available)

- **Training data:** `linxy/LaTeX_OCR` (configuration `small`) – image, LaTeX
- **Testing data:** 30 examples from the same dataset (`small`)

## 2. Evaluation Results

Metrics calculated on 30 test examples.

| Approach | Exact Match (%) | BLEU-4 |
|--------|----------------|--------|
| Zero-shot (no training) | 0.00 | 0.2625 |
| One-shot | 0.00 | 0.1209 |
| Fine-tuned (SFT on one dataset) | 0.00 | 0.2441 |

**Conclusions:** 
- Retraining does not significantly improve recognition quality: BLEU decreased by 0.02. 
- Exact Match did not increase from 0%, indicating that the model did not learn to generate correct LaTeX in most cases. 
- One-shot performed worse than zero-shot, likely due to poor example selection.

## 3. Демонстрация работы приложения

Screenshots confirming the functionality of the developed Streamlit application are located in `screenshots/` folder of the repository.

## 4. Conclusion

As part of this assignment, we successfully fine-tuned the VLM model for recognizing handwritten mathematical formulas. A web application was developed for demonstration. The metrics did not show significant improvement compared to the zero-shot approach. Future work could improve the quality by increasing the training data volume and adding augmentation.

## 5. Links

- repository: `https://github.com/ouvwte/LaTex-OCR-Huawei`
- Model (checkpoint): `https://huggingface.co/danilb575/Case_10_model`  
