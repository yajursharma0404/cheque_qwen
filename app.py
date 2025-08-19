import streamlit as st
from unsloth import FastVisionModel
import torch
from PIL import Image
import time
import re

# === Load the model ===
@st.cache_resource
def load_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "final_model",  # <-- adjust path if needed
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# === Helper function to extract field values ===
def extract_field(field_name, text):
    pattern = re.compile(rf"{field_name}\s*:\s*(.*?)(?=,\s*[\w\s]+?:|$)", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


# === App UI ===
st.set_page_config(page_title="Cheque Data Extraction", layout="wide")

st.title("🏦 Cheque Data Extraction App")

uploaded_file = st.file_uploader("📤 Upload a cheque image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show cheque image
    st.image(uploaded_file, caption="Uploaded Cheque", use_container_width=False, width=700)

    if st.button("Extract Data"):
        try:
            image = Image.open(uploaded_file).convert("RGB")

            context_prompt = (
                "Given image is of typical bank's cheque used in Indian banking system. "
                "Cheque image might also be multilingual (if it is not in english, it will be in devanagari script). "
                "Extract Account number, amount, bank name, IFSC code, payee name, cheque date "
                "(always written on the top right corner, ignore all other dates), and complete numeric code "
                "which is at the bottom middle of the image (it is written using magnetic ink) from the given image."
            )

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": context_prompt}
                ]}
            ]

            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            # Run inference
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=1.5,
                min_p=0.1,
            )
            duration = round(time.time() - start, 2)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # === Define fields to extract ===
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

            # Show fields 3 per row (excluding MICR Code)
            labels = list(fields.keys())
            for i in range(0, len(labels) - 1, 3):  # leave MICR Code last
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(labels) - 1:
                        label = labels[i + j]
                        value = extract_field(fields[label], output_text)
                        col.text_input(label, value=value, key=label)

            # MICR Code as a full-width text area
            micr_value = extract_field(fields["MICR Code"], output_text)
            st.text_area("MICR Code", value=micr_value, key="MICR Code", height=50)

            st.success(f"✅ Extraction complete in {duration} seconds!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
