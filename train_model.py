
import os
import torch
import pandas as pd
from datasets import Dataset
from PIL import Image
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Step 1: Load base model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Step 2: Add LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Step 3: Load dataset
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

# Step 4: Format dataset into conversation style
def convert_to_conversation(sample):
    try:
        image = Image.open(sample["image"]).convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample["prompt"]},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["response"]}],
                },
            ]
        }
    except Exception as e:
        print(f"Skipping {sample['image']} due to error: {e}")
        return None

raw_dataset = Dataset.from_dict({
    "image": df["image"].tolist(),
    "prompt": df["prompt"].tolist(),
    "response": df["response"].tolist(),
})

converted_dataset = []
for sample in raw_dataset:
    converted = convert_to_conversation(sample)
    if converted:
        converted_dataset.append(converted)

print(f"✅ Total usable samples: {len(converted_dataset)}")

# Step 5: Setup trainer
FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=converted_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=2,
        max_seq_length=2048,
    ),
)

# Step 6: Train the model
trainer.train()

# Step 7: Save LoRA adapters
print("💾 Saving LoRA adapters to lora_model/")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Step 8: Save full merged model
print("💾 Saving full merged model to final_model/")
FastVisionModel.for_inference(model)
model.save_pretrained_merged("final_model", tokenizer, safe_serialization=False)
