Training helper for LayoutLMv3

Files added:
- `train_layoutlmv3.py`: a script to finetune LayoutLMv3 on the JSONL produced by `build_layoutlm_dataset.py`.

Quick usage:

1. Create the dataset jsonl (already present if you ran `build_layoutlm_dataset.py`):

   - `authentiqa/data/layoutlm_token_dataset.jsonl`

2. Install dependencies (recommended in a venv):

   pip install -r requirements.txt

3. Run a short training run (example):

   python authentiqa/scripts/train_layoutlmv3.py --dataset_jsonl authentiqa/data/layoutlm_token_dataset.jsonl --do_train --epochs 1 --per_device_train_batch_size 2

Notes:
- The script automatically loads images using the `image_path` field in the jsonl.
- It derives label names from the `ner_tags_str` lists in the jsonl, so ensure they are present.
- For a production training run set up `TrainingArguments` (learning rate, steps, checkpointing) as desired.
