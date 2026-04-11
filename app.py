import streamlit as st
import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# web page settings
st.set_page_config(
    page_title="Huawei Future Star: LaTeX OCR",
    page_icon="📐",
    layout="wide"
)
st.title("📸 Recognition of handwritten formulas → LaTeX")
st.markdown("Upload a photo of the formula — the model will convert it to LaTeX and display the formula as an image.")

# model loading
@st.cache_resource
def load_fine_tuned_model():
    model_path = "./model_latex_only/final"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

try:
    processor, model, device = load_fine_tuned_model()
    st.success("✅ Model loaded successfully (Fine‑tuned SmolVLM)")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# predict validation*
def is_valid_latex(latex_str: str) -> bool:
    if not latex_str or len(latex_str) < 3:
        return False
    garbage_patterns = [
        r'\\documentclass', r'\\begin\{', r'\\end\{',
        r'(t12\.00)+', r'^\\$', r'^\\boxed\{\}$', r'^\\displaystyle$'
    ]
    for pat in garbage_patterns:
        if re.search(pat, latex_str, re.IGNORECASE):
            return False
    if not re.search(r'\\[a-zA-Z]+|[\^_{}]', latex_str):
        return False
    if len(set(latex_str)) < 3 and len(latex_str) > 20:
        return False
    return True

def clean_latex(raw: str) -> str:
    raw = raw.strip()
    raw = raw.replace("\\boxed{", "").replace("}", "")
    if raw in ["", "{}", "()"]:
        return ""
    if len(raw) > 500:
        raw = raw[:500] + "..."
    return raw

# main interface
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📂 Select an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None and st.button("🗑 Clear", use_container_width=True):
        uploaded_file = None
        if "last_prediction" in st.session_state:
            del st.session_state.last_prediction
        st.rerun()

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Your formula", use_container_width=True)
    else:
        st.info("⬅️ Upload an image")

# prediction, rendering
if uploaded_file is not None:
    if st.button("🔍 Recognize the formula", type="primary", use_container_width=True):
        with st.spinner("🧠 Analyzing the image..."):
            prompt = "Convert this handwritten formula to LaTeX code. Output only the LaTeX."
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512) #, do_sample=False
            generated = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            if "Assistant:" in generated:
                raw_latex = generated.split("Assistant:")[-1].strip()
            else:
                raw_latex = generated.strip()
            
            latex_code = clean_latex(raw_latex)
            valid = is_valid_latex(latex_code)
            
            st.session_state.last_prediction = {
                "latex": latex_code,
                "valid": valid,
                "raw": raw_latex
            }
    
    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        st.divider()
        
        if pred["valid"]:
            st.subheader("📜 Recognized LaTeX‑код")
            st.code(pred["latex"], language="latex")
            
            st.subheader("📐 Visualization of the formula")
            try:
                fig, ax = plt.subplots(figsize=(7, 2))
                ax.text(0.5, 0.5, f"${pred['raw']}$", fontsize=20, ha='center', va='center') # 
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Failed to render formula: {e}")
                st.code(pred["latex"])
            
            st.download_button(
                label="📋 Copy LaTeX",
                data=pred["latex"],
                file_name="formula.tex",
                mime="text/plain",
                use_container_width=True
            )
            
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append(pred["latex"])
        else:
            st.error("❌ The model failed to recognize the formula. Possible reasons:")
            st.markdown("""
            - **Poor photo quality** (blurry, shadows, glare)
            - **Formula too complex** (model trained on limited data)
            - **Unusual handwriting** or slant
            """)
            st.info("Try taking a sharper photo, increasing the contrast, or simplifying the formula.")
            if pred["latex"] and len(pred["latex"]) > 5:
                with st.expander("See what the model generated for debugging"):
                    st.code(pred["raw"])
            del st.session_state.last_prediction
            st.rerun()

# bottom menu
st.divider()
tab1, tab2, tab3 = st.tabs(["📖 About project", "📊 Results", "🛠 Technical details"])

with tab1:
    st.markdown("""
    **Task:** Recognition of handwritten mathematical formulas and conversion into LaTeX.
    
    **Model:** `HuggingFaceTB/SmolVLM-256M-Instruct` — light Vision‑Language model, retrained using the LoRA method on the dataset `linxy/LaTeX_OCR`.
    
    **How to use:**
    1. Upload a clear photo of the formula.
    2. Click "🔍 Recognize the formula"
    3. Get the LaTeX code and its visualization as an image.
    4. If an error occurs, try a different photo.
    
    This project was completed as part of the qualifying assignment for the Huawei Future Star 2026 internship in the Multimodal Reasoning for STEM track.
    
    Developer: Brunov Daniil ([e-mail](danilabrunov@gmail.com)).            
    """)

with tab2:
    st.markdown("**Quality comparison:**")
    st.dataframe({
        "Метод": ["Zero‑shot", "One‑shot", "Fine‑tuned (SFT)"],
        "Exact Match": ["0.00%", "0.00%", "0.0%"],
        "BLEU-4": ["0.2625", "0.1209", "0.2441"]
    }, use_container_width=True)
    st.caption("BLEU is calculated symbol-by-symbol. The fine-tuned model is slightly inferior to the baseline model. " \
                "This performance is due to the small volume of training data due to hardware limitations.")

with tab3:
    st.markdown("""
    - **Architecture:** SmolVLM-256M-Instruct (Vision Encoder + Language Model)
    - **Training:** LoRA (r=8, alpha=16), 3 epochs, learning rate 2e-4
    - **Data:** `linxy/LaTeX_OCR` (small)
    - **Libraries:** PyTorch, Transformers, Streamlit, Matplotlib
    - **Hardware:** GPU (CUDA) / CPU
    """)