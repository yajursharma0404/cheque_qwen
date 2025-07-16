from unsloth import FastVisionModel
import torch
import os
from transformers import TextStreamer
from PIL import Image
import time



# === Load the model (merged final model) ===
model_path = "/mnt/d/Projects/Qwen Colab Notebook/Qwen v3/final_model"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_path,
    load_in_4bit=False,
)
FastVisionModel.for_inference(model)

# === Setup for inference ===
image_folder = "/mnt/d/Projects/test_data/test_images"

context_prompt = (
    "Given image is of typical bank's cheque used in Indian banking system. "
    "Cheque image might also be multilingual (if it is not in english, it will be in devanagari script). "
    "Extract Account number, amount, bank name, IFSC code, Pincode, payee name, cheque date "
    "(always written on the top right corner, ignore all other dates), and complete numeric code "
    "which is at the bottom middle of the image (it is written using magnetic ink) from the given image."
)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]

# === Run inference ===
for image_name in image_files:
    try:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

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

        # Start timer
        start = time.time()

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )
        # End timer
        duration = round(time.time() - start, 2)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n🧾 Output for {image_name}:\n{output_text}")
        print(f"⏱ Time taken: {duration} seconds")

    except Exception as e:
        print(f"\n❌ Error on {image_name}: {e}")
