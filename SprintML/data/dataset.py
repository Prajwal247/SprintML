from datasets import load_dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH, TASK_INFO


def load_glue_datasets() -> dict:
    """Load SST-2, QNLI, MNLI, and QQP from the GLUE benchmark."""
    print("Loading datasets...")
    datasets = {
        "sst2": load_dataset("glue", "sst2"),
        "qnli": load_dataset("glue", "qnli"),
        "mnli": load_dataset("glue", "mnli"),
        "qqp": load_dataset("glue", "qqp"),
    }

    for name, dataset in datasets.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  Train: {len(dataset['train'])} examples")
        if "validation" in dataset:
            print(f"  Validation: {len(dataset['validation'])} examples")
        if "validation_matched" in dataset:
            print(f"  Validation (matched): {len(dataset['validation_matched'])} examples")
            print(
                f"  Validation (mismatched): {len(dataset['validation_mismatched'])} examples"
            )

    return datasets


def _preprocess_function(examples, dataset_name: str, tokenizer):
    info = TASK_INFO[dataset_name]
    sentence_keys = info["sentence_keys"]

    if len(sentence_keys) == 1:
        result = tokenizer(
            examples[sentence_keys[0]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    else:
        result = tokenizer(
            examples[sentence_keys[0]],
            examples[sentence_keys[1]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    result["labels"] = examples[info["label_column"]]
    return result


def preprocess_datasets(raw_datasets: dict) -> dict:
    """Tokenize and format all datasets for PyTorch training."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}

    tokenized = {}
    for name, dataset in raw_datasets.items():
        print(f"Tokenizing {name}...")
        tokenized[name] = {}

        # Tokenize train split
        cols_to_remove = [c for c in dataset["train"].column_names if c not in keep_cols]
        tokenized[name]["train"] = dataset["train"].map(
            lambda x: _preprocess_function(x, name, tokenizer),
            batched=True,
            remove_columns=cols_to_remove,
        )

        # Tokenize validation split (MNLI uses validation_matched)
        val_key = "validation_matched" if name == "mnli" else "validation"
        val_cols_to_remove = [
            c for c in dataset[val_key].column_names if c not in keep_cols
        ]
        tokenized[name]["validation"] = dataset[val_key].map(
            lambda x: _preprocess_function(x, name, tokenizer),
            batched=True,
            remove_columns=val_cols_to_remove,
        )

        # Set torch format
        available = tokenized[name]["train"].column_names
        fmt_cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in available:
            fmt_cols.append("token_type_ids")

        tokenized[name]["train"].set_format(type="torch", columns=fmt_cols)
        tokenized[name]["validation"].set_format(type="torch", columns=fmt_cols)

    print("\nPreprocessing complete!")
    return tokenized
