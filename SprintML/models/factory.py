from transformers import AutoModelForSequenceClassification

from config import MODEL_NAME, HYPERPARAMS, TASK_INFO
from models.soft_prompt import SoftPromptModel
from models.prefix_tuning import PrefixTuningModel
from models.lora import create_lora_model, create_full_finetuning_model
from models.single_layer import SingleLayerModel
from models.soft_prompt_lora import SoftPromptLoRAModel
from models.prefix_lora import PrefixLoRAModel


def create_model(method: str, dataset_name: str):
    """Instantiate the appropriate model for a given method and dataset."""
    num_labels = TASK_INFO[dataset_name]["num_labels"]
    params = HYPERPARAMS[method]

    if method == "soft_prompt":
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, attn_implementation="eager"
        )
        return SoftPromptModel(base, num_labels, params["prompt_length"])

    if method == "prefix":
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, attn_implementation="eager"
        )
        return PrefixTuningModel(base, num_labels, params["prefix_length"])

    if method == "full_finetuning":
        return create_full_finetuning_model(MODEL_NAME, num_labels)

    if method == "lora":
        return create_lora_model(MODEL_NAME, num_labels, params["lora_r"])

    if method == "single_layer":
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, attn_implementation="eager"
        )
        return SingleLayerModel(base, num_labels)

    if method == "soft_prompt_lora":
        return SoftPromptLoRAModel(
            MODEL_NAME, num_labels, params["prompt_length"], params["lora_r"]
        )

    if method == "prefix_lora":
        return PrefixLoRAModel(
            MODEL_NAME, num_labels, params["prefix_length"], params["lora_r"]
        )

    raise ValueError(f"Unknown method: {method}")
