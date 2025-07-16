# Cheque Information Extraction with Qwen Vision

This project trains and uses a vision-language model to extract structured data from Indian bank cheque images using Unsloth's Qwen2-VL-2B model.

## 📁 Folder Structure

- `train_images/` – Images used for training
- `test_images/` – Images used for inference
- `cheque_info.csv` – CSV file containing labels for training
- `train_model.py` – Script to train the model
- `run_model.py` – Script to run inference on cheque images

Usage

1. Install dependencies

```bash
pip install -r requirements.txt

2. Train the model

python train_model.py

3. Run inference

python run_model.py
