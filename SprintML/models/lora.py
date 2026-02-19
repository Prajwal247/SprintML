from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def create_lora_model(
    base_model_name: str,
    num_labels: int,
    lora_r: int = 8,
    lora_alpha: int = 16,
):
    """Create a model with LoRA applied to the Q and V attention matrices."""
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels, attn_implementation="eager"
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model


def create_full_finetuning_model(base_model_name: str, num_labels: int):
    """Load a standard model with all parameters trainable."""
    return AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels, attn_implementation="eager"
    )
