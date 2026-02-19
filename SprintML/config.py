MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 128
SEED = 42

# Differential privacy parameters
EPSILON = 8.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

METHODS = [
    "soft_prompt",
    "prefix",
    "full_finetuning",
    "lora",
    "single_layer",
    "soft_prompt_lora",
    "prefix_lora",
]

DATASETS = ["sst2", "qnli", "mnli", "qqp"]

HYPERPARAMS = {
    "soft_prompt": {
        "learning_rate": 5e-3,
        "prompt_length": 10,
        "epochs": 5,
        "batch_size": 32,
    },
    "prefix": {
        "learning_rate": 5e-3,
        "prefix_length": 10,
        "epochs": 5,
        "batch_size": 32,
    },
    "full_finetuning": {
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 32,
    },
    "lora": {
        "learning_rate": 3e-4,
        "lora_r": 8,
        "epochs": 3,
        "batch_size": 32,
    },
    "single_layer": {
        "learning_rate": 1e-3,
        "epochs": 5,
        "batch_size": 32,
    },
    "soft_prompt_lora": {
        "learning_rate": 5e-4,
        "prompt_length": 10,
        "lora_r": 8,
        "epochs": 5,
        "batch_size": 32,
    },
    "prefix_lora": {
        "learning_rate": 5e-4,
        "prefix_length": 10,
        "lora_r": 8,
        "epochs": 5,
        "batch_size": 32,
    },
}

TASK_INFO = {
    "sst2": {"sentence_keys": ["sentence"], "num_labels": 2, "label_column": "label"},
    "qnli": {
        "sentence_keys": ["question", "sentence"],
        "num_labels": 2,
        "label_column": "label",
    },
    "mnli": {
        "sentence_keys": ["premise", "hypothesis"],
        "num_labels": 3,
        "label_column": "label",
    },
    "qqp": {
        "sentence_keys": ["question1", "question2"],
        "num_labels": 2,
        "label_column": "label",
    },
}
