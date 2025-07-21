from unsloth import FastVisionModel, is_bf16_supported
import os
import pandas as pd
from datasets import Dataset
from PIL import Image
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import torch

# Load the base Qwen2.5 3B VL Instruct model in 4-bit for efficient training
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Add LoRA adapter configuration
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing = "unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load the cheque info CSV and prepare prompt/image columns
df = pd.read_csv("cheque_info.csv")
df["filename"] = df["path"].apply(lambda x: os.path.basename(str(x).replace("\\", "/")))
df["image"] = df["filename"].apply(lambda x: os.path.join("train_images", x))
df["prompt"] = (
    "Given image is of typical bank's cheque used in Indian banking system. "
    "Cheque image might also be multilingual (if it is not in english, it will be in devanagari script). "
    "Extract Account number, amount, bank name, IFSC code, Pincode, payee name, cheque date (always written on the top right corner, ignore all other dates), "
    "and complete numeric code which is at the bottom middle of the image (it is written using magnetic ink) from the given image."
)
df["response"] = df["text"]

# Convert to proper message format
def convert_to_conversation(sample):
    try:
        image = Image.open(sample["image"]).convert("RGB")
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": sample["prompt"]},
                    {"type": "image", "image": image},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["response"]}
                ]},
            ]
        }
    except Exception as e:
        print(f"⚠️ Skipping {sample['image']} due to error: {e}")
        return None

# Create dataset
raw_dataset = Dataset.from_dict({
    "image": df["image"].tolist(),
    "prompt": df["prompt"].tolist(),
    "response": df["response"].tolist(),
})

converted_dataset = []
for sample in raw_dataset:
    result = convert_to_conversation(sample)
    if result:
        converted_dataset.append(result)

print(f"✅ Loaded {len(converted_dataset)} usable samples.")

# Enable training mode
FastVisionModel.for_training(model)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=converted_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    packing = False,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs_qwen2_5",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        dataset_num_proc=2,
    ),
)

# Start training
trainer.train()

# Save LoRA adapters
#print("💾 Saving LoRA adapters to lora_model_qwen2_5/")
#model.save_pretrained("lora_model_qwen2_5")
#tokenizer.save_pretrained("lora_model_qwen2_5")

# Save merged model
print("💾 Saving full model to final_model_qwen2_5/")
FastVisionModel.for_inference(model)
model.save_pretrained_merged("final_model_qwen2_5", tokenizer, safe_serialization=False)
