"""
Fine-tuning opcional com Hugging Face Trainer.
Dataset esperado em data/datasets/ (CSV ou JSONL) com coluna 'text' (entrada) e 'target' (saída markdown/quiz).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from ..core.config import settings


def load_local_dataset(path: str | Path) -> Dataset:
    path = str(path)
    if path.endswith(".csv"):
        return load_dataset("csv", data_files=path)["train"]
    if path.endswith(".jsonl") or path.endswith(".json"):
        return load_dataset("json", data_files=path)["train"]
    raise ValueError("Formato de dataset não suportado (use .csv, .json, .jsonl).")


def tokenize_function(examples, tok, max_len=1024):
    model_inputs = tok(examples["text"], max_length=max_len, truncation=True)
    labels = tok(examples["target"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(
    dataset_path: str = "data/datasets/train.jsonl",
    output_dir: str = "models/finetuned",
):
    tok = AutoTokenizer.from_pretrained(
        settings.hf_model_name, use_auth_token=settings.hf_token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.hf_model_name, use_auth_token=settings.hf_token
    )

    ds = load_local_dataset(dataset_path)
    tokenized = ds.map(
        lambda x: tokenize_function(x, tok),
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        fp16=False,
        save_total_limit=2,
        logging_steps=20,
        save_steps=200,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
