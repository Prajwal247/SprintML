import math

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, default_collate
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from config import EPSILON, DELTA, MAX_GRAD_NORM
from training.dp_utils import fix_model_for_opacus
from models.soft_prompt import SoftPromptModel
from models.prefix_tuning import PrefixTuningModel
from models.single_layer import SingleLayerModel
from models.soft_prompt_lora import SoftPromptLoRAModel
from models.prefix_lora import PrefixLoRAModel

CUSTOM_WRAPPER_TYPES = (
    SoftPromptModel,
    PrefixTuningModel,
    SingleLayerModel,
    SoftPromptLoRAModel,
    PrefixLoRAModel,
)


def _custom_collate_fn(batch):
    batch_dict = default_collate(batch)
    if "attention_mask" in batch_dict:
        batch_dict["attention_mask"] = batch_dict["attention_mask"].float()
    return batch_dict


def _train_epoch(model, dataloader, optimizer, scheduler, device,
                 use_manual_dp=False, sigma=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        try:
            outputs = model(**batch)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                filtered = {k: v for k, v in batch.items()
                            if k in ("input_ids", "attention_mask", "labels")}
                outputs = model(**filtered)
            else:
                raise

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if use_manual_dp and sigma is not None:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=sigma * max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device,
                    )
                    param.grad += noise

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def _evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            try:
                outputs = model(**batch)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    filtered = {k: v for k, v in batch.items()
                                if k in ("input_ids", "attention_mask", "labels")}
                    outputs = model(**filtered)
                else:
                    raise

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return {"accuracy": accuracy, "loss": total_loss / len(dataloader)}


def train_model(model, train_dataset, val_dataset, device,
                num_epochs=3, batch_size=32, learning_rate=5e-4):
    """Full training pipeline with differential privacy support."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=_custom_collate_fn,
    )

    fix_model_for_opacus(model)

    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            param.requires_grad = False

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    is_custom = isinstance(model, CUSTOM_WRAPPER_TYPES)

    if is_custom:
        sigma = math.sqrt(2 * math.log(1.25 / DELTA)) / EPSILON
        use_manual_dp = True
        privacy_engine = None
        print(f"  Manual DP enabled: eps={EPSILON}, delta={DELTA}, sigma={sigma:.4f}, C={MAX_GRAD_NORM}")
    else:
        from opacus import PrivacyEngine
        use_manual_dp = False
        sigma = None
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=num_epochs,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"  Opacus DP enabled: eps={EPSILON}, delta={DELTA}, C={MAX_GRAD_NORM}")

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_accuracy = 0.0
    results = []

    print(f"\nTraining Configuration:")
    print(f"  Total training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num epochs: {num_epochs}")

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_manual_dp=use_manual_dp, sigma=sigma, max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = _evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy'] * 100:.2f}%)")

        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            print(f"  New best accuracy: {best_accuracy:.4f}")

        if privacy_engine is not None:
            eps_spent = privacy_engine.get_epsilon(DELTA)
            print(f"  Privacy spent: eps = {eps_spent:.2f}")
        else:
            print(f"  Privacy budget: eps = {EPSILON} (manual DP)")

    print(f"\n{'=' * 50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print(f"{'=' * 50}")

    return results, best_accuracy
