"""
SprintML — Comparing prompt-learning methods on TinyBERT with differential privacy.

Runs all 7 adaptation methods across 4 GLUE datasets and saves results to CSV.
"""

import warnings

import torch
import pandas as pd

from config import METHODS, DATASETS, HYPERPARAMS
from utils import set_seed, get_device, count_parameters
from data.dataset import load_glue_datasets, preprocess_datasets
from models.factory import create_model
from training.trainer import train_model

warnings.filterwarnings("ignore")


def main():
    set_seed()
    device = get_device()

    raw_datasets = load_glue_datasets()
    tokenized_datasets = preprocess_datasets(raw_datasets)

    all_results: dict = {}

    for dataset_name in DATASETS:
        print(f"\n{'-' * 100}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'-' * 100}\n")

        all_results[dataset_name] = {}
        train_data = tokenized_datasets[dataset_name]["train"]
        val_data = tokenized_datasets[dataset_name]["validation"]

        for method in METHODS:
            print(f"\n{'-' * 100}")
            print(f"Method: {method.upper()}")
            print(f"{'-' * 100}")

            try:
                model = create_model(method, dataset_name)

                if hasattr(model, "count_parameters"):
                    trainable_params = model.count_parameters()
                else:
                    trainable_params, _ = count_parameters(model)

                print(f"Trainable parameters: {trainable_params:,}")

                params = HYPERPARAMS[method]
                history, best_acc = train_model(
                    model,
                    train_data,
                    val_data,
                    device=device,
                    num_epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    learning_rate=params["learning_rate"],
                )

                all_results[dataset_name][method] = {
                    "best_accuracy": best_acc,
                    "trainable_params": trainable_params,
                    "history": history,
                }
                print(f"\nBest Accuracy: {best_acc:.4f}")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError training {method} on {dataset_name}: {e}")
                all_results[dataset_name][method] = {
                    "best_accuracy": 0,
                    "trainable_params": 0,
                    "error": str(e),
                }

    # Compile results into a table
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            if dataset in all_results and method in all_results[dataset]:
                result = all_results[dataset][method]
                rows.append({
                    "Dataset": dataset.upper(),
                    "Method": method.replace("_", " ").title(),
                    "Accuracy (%)": f"{result['best_accuracy'] * 100:.2f}",
                    "Trainable Params": f"{result['trainable_params']:,}",
                })

    results_df = pd.DataFrame(rows)

    print(f"\n{'=' * 100}")
    print("FINAL RESULTS")
    print(f"{'=' * 100}\n")
    print(results_df.to_string(index=False))

    output_path = "results_epsilon_8.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
