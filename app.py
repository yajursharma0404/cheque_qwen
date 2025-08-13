#"streamlit run app.py" in WSL terminal 

import streamlit as st
from PIL import Image
import torch
import os
from unsloth import FastVisionModel
from transformers import TextStreamer
import time

# === Load model ===
@st.cache_resource
def load_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="final_model",  # Path to your merged model
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# === Prompt ===
CONTEXT_PROMPT = (
    "Given image is of typical bank's cheque used in Indian banking system. "
    "Cheque image might also be multilingual (if it is not in english, it will be in devanagari script). "
    "Extract Account number, amount, bank name, IFSC code, Pincode, payee name, cheque date "
    "(always written on the top right corner, ignore all other dates), and complete numeric code "
    "which is at the bottom middle of the image (it is written using magnetic ink) from the given image."
)

# === Streamlit UI ===
st.title("🧾 Cheque Information Extractor")
uploaded_file = st.file_uploader("Upload a cheque image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Cheque", use_column_width=True)

    with st.spinner("🔍 Extracting information..."):
        start_time = time.time()

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CONTEXT_PROMPT}
            ]}
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            min_p=0.1,
        )

        end_time = time.time()

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.success(f"✅ Inference completed in {end_time - start_time:.2f} seconds.")

    # === Show raw model output ===
    st.subheader("📝 Raw Model Output")
    st.text_area("Model Output", value=output_text, height=200)

    # === Parse output text ===
    import re

    def extract_field(field_name, text):
        pattern = re.compile(rf"{field_name}\s*:\s*(.*?)(?=,\s*[\w\s]+?:|$)", re.IGNORECASE)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    # Field mappings from model output
    fields = {
        "Account Number": "account_number",
        "Amount": "amount",
        "Bank Name": "bank_name",
        "IFSC Code": "ifs_code",
        "Payee Name": "payee_name",
        "Cheque Date": "cheque_date",
        "MICR Code": "MICR code"
    }

    st.subheader("📋 Extracted Information")
    for label, key in fields.items():
        value = extract_field(key, output_text)
        st.text_input(label, value=value, key=label)
