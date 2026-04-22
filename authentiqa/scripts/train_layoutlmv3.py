import argparse
import json
from pathlib import Path
from collections import Counter

from PIL import Image
import numpy as np

from datasets import load_dataset
import torch

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)


def get_label_list_from_jsonl(jsonl_path):
    labels = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for lab in rec.get("ner_tags_str", []):
                labels.add(lab)

    # Ensure 'O' is first if present for convention
    labels = sorted(labels)
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    return labels


def prepare_dataset(dataset, image_column="image_path"):
    # Add image PIL objects in a map step
    def _load_image(example):
        img_path = example[image_column]
        example["image"] = Image.open(img_path).convert("RGB")
        # bboxes are already normalized to 0-1000 ints
        return example

    return dataset.map(_load_image)


class DataCollatorForLayoutLMv3:
    def __init__(self, processor: LayoutLMv3Processor):
        self.processor = processor

    def __call__(self, features):
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["bboxes"] for f in features]
        labels = [f["ner_tags"] for f in features]

        # Use 'text' arg (processor expects text=...) and provide word_labels so
        # the processor can create token-aligned labels (-100 for subword tokens)
        encoding = self.processor(
            images=images,
            text=words,
            boxes=boxes,
            word_labels=labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return encoding


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for p, l in zip(predictions, labels):
        for pred_id, label_id in zip(p, l):
            if label_id != -100:
                true_predictions.append(pred_id)
                true_labels.append(label_id)

    if len(true_labels) == 0:
        return {"accuracy": 0.0}

    acc = (np.array(true_predictions) == np.array(true_labels)).mean()
    return {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./layoutlmv3-finetuned")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Run a one-batch forward pass and exit")
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Dataset jsonl not found: {dataset_jsonl}")

    label_list = get_label_list_from_jsonl(dataset_jsonl)
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    # Load the JSONL with datasets
    dataset = load_dataset("json", data_files=str(dataset_jsonl))
    # dataset['train'] exists even if a single split
    split = list(dataset.keys())[0]
    ds = dataset[split]

    ds = prepare_dataset(ds, image_column="image_path")

    # We have pre-extracted OCR boxes in the dataset, so disable the processor's
    # internal OCR (apply_ocr=False). This allows passing boxes/text directly.
    processor = LayoutLMv3Processor.from_pretrained(args.model_name_or_path, apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForLayoutLMv3(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps" if args.do_eval else "no",
        save_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=200,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds if args.do_train else None,
        eval_dataset=ds if args.do_eval else None,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if args.do_eval else None,
    )

    # Quick record validation helper
    def validate_record(rec):
        words = rec.get("words", [])
        boxes = rec.get("bboxes", [])
        labels = rec.get("ner_tags", [])
        if not (len(words) == len(boxes) == len(labels)):
            return False, f"length mismatch: words={len(words)}, boxes={len(boxes)}, labels={len(labels)}"
        # check boxes in 0-1000 range
        for b in boxes:
            if any((v is None) for v in b):
                return False, "box contains None"
            # allow some negative due to OCR artifacts but warn
            if any((not isinstance(v, (int, float))) for v in b):
                return False, "box coordinate not numeric"
        return True, "ok"

    if args.smoke_test:
        # Run a single forward pass on a tiny batch to validate end-to-end pipeline
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # pick first two examples (or less)
        sample_count = min(2, len(ds))
        examples = [ds[i] for i in range(sample_count)]

        # validate examples
        for ex in examples:
            ok, msg = validate_record(ex)
            if not ok:
                raise ValueError(f"Record validation failed: {msg} | rec id={ex.get('id')}")

        batch = data_collator(examples)

        # move tensors to device
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        # forward
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits if hasattr(outputs, "logits") else None
        loss = outputs.loss if hasattr(outputs, "loss") else None
        print("Smoke forward pass results:")
        if logits is not None:
            print("logits shape:", logits.shape)
        if loss is not None:
            print("loss:", float(loss))
        print("Smoke test done.")
        return

    if args.do_train:
        trainer.train()

    if args.do_eval:
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
