"""Train the improved proposed neural model on the prepared stellar dataset.

References
----------
PyTorch CrossEntropyLoss:
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

PyTorch AdamW:
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

PyTorch ReduceLROnPlateau:
https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

PyTorch torch.save:
https://docs.pytorch.org/docs/stable/generated/torch.save.html

scikit-learn StandardScaler:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

scikit-learn balanced_accuracy_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

scikit-learn f1_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

scikit-learn SimpleImputer:
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

pandas DataFrame.to_csv:
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

pandas read_csv:
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

Matplotlib pyplot.plot:
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

Matplotlib pyplot.savefig:
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import TabularResNetClassifier
from src.reproducibility import PROJECT_SEED, configure_determinism, seed_worker


RANDOM_STATE = PROJECT_SEED


@dataclass(frozen=True)
class TrainingConfig:
    """Training and architecture settings for one proposed-model stage."""

    experiment: str = "tuned"
    batch_size: int = 256
    epochs: int = 45
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    block_hidden_dim: int = 512
    num_blocks: int = 4
    dropout: float = 0.05
    gradient_clip_norm: float | None = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 4
    early_stopping_patience: int | None = 100
    min_delta: float = 1e-4
    use_scheduler: bool = True


INITIAL_CONFIG = TrainingConfig(
    experiment="initial",
    epochs=30,
    learning_rate=1e-3,
    dropout=0.10,
    gradient_clip_norm=None,
    early_stopping_patience=None,
    min_delta=0.0,
    use_scheduler=False,
)
TUNED_CONFIG = TrainingConfig()
CONFIG = TUNED_CONFIG

TARGET_COLUMNS = ["class", "class_encoded"]
HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "balanced_accuracy",
    "macro_f1",
    "learning_rate",
]


def save_training_history(
    history_records: list[dict[str, int | float]], history_path: Path
) -> pd.DataFrame:
    """Save epoch metrics and return the values read from the saved CSV."""
    history = pd.DataFrame(history_records, columns=HISTORY_COLUMNS)
    history.to_csv(history_path, index=False)
    return pd.read_csv(history_path)


def save_training_plots(
    history: pd.DataFrame, figures_dir: Path, file_prefix: str = "tabular_resnet"
) -> None:
    """Create loss and validation-metric figures from saved training history."""
    loss_figure, loss_axis = plt.subplots()
    loss_axis.plot(history["epoch"], history["train_loss"], label="Training loss")
    loss_axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Tabular ResNet Training and Validation Loss")
    loss_axis.legend()
    loss_figure.savefig(
        figures_dir / f"{file_prefix}_loss.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(loss_figure)

    balanced_accuracy_figure, balanced_accuracy_axis = plt.subplots()
    balanced_accuracy_axis.plot(history["epoch"], history["balanced_accuracy"])
    balanced_accuracy_axis.set_xlabel("Epoch")
    balanced_accuracy_axis.set_ylabel("Validation balanced accuracy")
    balanced_accuracy_axis.set_title("Tabular ResNet Validation Balanced Accuracy")
    balanced_accuracy_figure.savefig(
        figures_dir / f"{file_prefix}_balanced_accuracy.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(balanced_accuracy_figure)

    macro_f1_figure, macro_f1_axis = plt.subplots()
    macro_f1_axis.plot(history["epoch"], history["macro_f1"])
    macro_f1_axis.set_xlabel("Epoch")
    macro_f1_axis.set_ylabel("Validation macro F1")
    macro_f1_axis.set_title("Tabular ResNet Validation Macro F1")
    macro_f1_figure.savefig(
        figures_dir / f"{file_prefix}_macro_f1.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(macro_f1_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("initial", "tuned"),
        default="tuned",
        help="Train the historical initial configuration or the tuned configuration.",
    )
    args = parser.parse_args()
    config = INITIAL_CONFIG if args.experiment == "initial" else TUNED_CONFIG
    generator = configure_determinism(RANDOM_STATE)

    project_root = Path(__file__).resolve().parents[1]

    train = pd.read_csv(project_root / "data" / "train.csv")
    val = pd.read_csv(project_root / "data" / "val.csv")

    # Separate features and targets.
    X_train = train.drop(columns=TARGET_COLUMNS)
    y_train = train["class_encoded"]

    X_val = val.drop(columns=TARGET_COLUMNS)
    y_val = val["class_encoded"]

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Train and validation feature columns do not match.")

    print(f"Features: {X_train.shape[1]}")
    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    # Fit preprocessing ONLY on train.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_np = imputer.fit_transform(X_train)
    X_val_np = imputer.transform(X_val)

    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np = scaler.transform(X_val_np)

    train_dataset = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val_np, dtype=torch.float32),
        torch.tensor(y_val.to_numpy(), dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # CPU is intentional: it is the canonical cross-platform reproduction target.
    device = torch.device("cpu")

    print("Model: TabularResNetClassifier")
    print(f"Device: {device}")

    model = TabularResNetClassifier(
        num_features=X_train.shape[1],
        num_classes=3,
        hidden_dim=config.hidden_dim,
        block_hidden_dim=config.block_hidden_dim,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_factor,
            patience=config.lr_patience,
        )
        if config.use_scheduler
        else None
    )

    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_macro_f1 = 0.0
    best_val_loss = 0.0
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "tabular_resnet_tuned" if config.experiment == "tuned" else "tabular_resnet"
    best_path = models_dir / f"{file_prefix}.pt"
    training_reports_dir = project_root / "reports" / "training"
    training_reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = training_reports_dir / f"{file_prefix}_history.csv"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    history_records: list[dict[str, int | float]] = []

    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()

        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, targets)

            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()

        val_loss = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                logits = model(features)
                loss = criterion(logits, targets)

                val_loss += loss.item() * features.size(0)

                preds = logits.argmax(dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        balanced_acc = balanced_accuracy_score(
            true_labels,
            predictions,
        )

        macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
        )

        history_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "balanced_accuracy": balanced_acc,
                "macro_f1": macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"balanced_acc={balanced_acc:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if balanced_acc > best_balanced_accuracy + config.min_delta:
            best_balanced_accuracy = balanced_acc
            best_epoch = epoch
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_path,
            )

            print(
                f"  -> saved new best model "
                f"(balanced_acc={balanced_acc:.4f})"
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(balanced_acc)

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} "
                "epochs without validation balanced-accuracy improvement."
            )
            break

    saved_history = save_training_history(history_records, history_path)
    save_training_plots(saved_history, figures_dir, file_prefix=file_prefix)

    print()
    print("Model: TabularResNetClassifier")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation balanced accuracy: {best_balanced_accuracy:.4f}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_path.relative_to(project_root)}")
    print(f"Training history saved to: {history_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
