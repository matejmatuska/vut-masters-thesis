import argparse
import os
import time

import mlflow
import optuna
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.transforms import NormalizeFeatures

import utils
from train_common import (
    evaluate,
    get_allowed_models,
    get_dataset_factory,
    get_model_factory,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "model", type=str, choices=get_allowed_models(), help="Model to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run training on",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training and testing"
    )

    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--tolerance", type=float, default=5e-3, help="Early stopping tolerance"
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="xmatus36-gnns",
        help="MLflow experiment name",
    )
    parser.add_argument("--mlflow-run", type=str, help="MLflow main run name")
    return parser.parse_args()


args = parse_args()
print("Model:", args.model)

which = args.model.split("-")

dset_factory = get_dataset_factory(which[0])
train_set = dset_factory(args.dataset_path, "train", transform=NormalizeFeatures())
val_set = dset_factory(args.dataset_path, "val", transform=NormalizeFeatures())
test_set = dset_factory(args.dataset_path, "test", transform=NormalizeFeatures())

print(
    f"Dataset samples: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
)

print("First train sample:", train_set[0])
print("First train sample:", train_set[1])
print("First train sample:", train_set[2])
print("First train sample:", train_set[3])
print("First train sample:", train_set[4])
print("Lbael map:", train_set.label_map)

num_classes = train_set.num_classes

device = torch.device(args.device)
class_weights = utils.compute_class_weights(train_set).to(device)

print(f"Number of classes: {num_classes}")
print("Class Weights:", class_weights)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model_factory = get_model_factory(which)

os.makedirs("models", exist_ok=True)
model_dir = os.path.join("models")
os.makedirs(model_dir, exist_ok=True)


def objective(trial: optuna.Trial) -> float:
    if which[0] == "baseline":
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_int("hidden_dim", 96, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    elif which[0] == "repr1":
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_int("hidden_dim", 96, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    elif which[0] == "repr2":
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_int("hidden_dim", 96, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = model_factory(
        train_set,
        hidden_dim=hidden_dim,
        port_dim=2,
        dropout=dropout,
        nlayers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = utils.EarlyStopping(
        patience=args.patience, tolerance=args.tolerance
    )

    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "model": args.model,
                "dataset": args.dataset_path,
                "device": args.device,
                "batch_size": args.batch_size,
            }
        )
        mlflow.log_params(
            {
                "num_classes": num_classes,
                "lr": learning_rate,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "epochs": args.epochs,
            }
        )
        # mlflow.log_params(vars(args))

        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1, all_preds, all_labels = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                },
                step=epoch,
            )

            trial.report(val_f1, step=epoch - 1)  # pruner assumes start at 0

            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch} with val_f1: {val_f1}")
                raise optuna.exceptions.TrialPruned()

            log(Epoch=epoch, Loss=train_loss, Val=val_loss, Acc=val_acc, F1=val_f1)
            times.append(time.time() - start)

            if early_stopping(
                val_loss,
                model,
                val_f1=val_f1,
                epoch=epoch,
                all_preds=all_preds,
                all_labels=all_labels,
                val_loss=val_loss,
                val_acc=val_acc,
                train_loss=train_loss,
            ):
                print(
                    f"Early stopping at epoch {epoch} with min val_loss: {early_stopping.best_loss}"
                )
                mlflow.log_metric("reached_epoch", epoch)
                break

        if early_stopping:
            all_preds = early_stopping.data_at_best["all_preds"]
            all_labels = early_stopping.data_at_best["all_labels"]
            val_loss = early_stopping.data_at_best["val_loss"]
            val_acc = early_stopping.data_at_best["val_acc"]
            val_f1 = early_stopping.data_at_best["val_f1"]
            train_loss = early_stopping.data_at_best["val_loss"]
            utils.log_class_stats(all_preds, all_labels, suffix="val")
        else:
            # only log confusion matrix and classification report after the last epoch
            utils.log_class_stats(all_preds, all_labels, suffix="val")

        mlflow.log_metrics(
            {
                "real_train_loss": train_loss,
                "real_val_loss": val_loss,
                "real_accuracy": val_acc,
                "real_f1": val_f1,
            },
            step=epoch,
        )

        mlflow.log_metric("median_epoch_time", torch.tensor(times).median())

        model_state = (
            early_stopping.best_model_state if early_stopping else model.state_dict()
        )
        model_path = os.path.join(model_dir, f"model_{trial.number}.pth")
        torch.save(model_state, model_path)
        trial.set_user_attr("model_path", model_path)

    retf1 = early_stopping.data_at_best["val_f1"] if early_stopping else val_f1
    print(f"objective returning with val_f1 {retf1}")
    return early_stopping.data_at_best["val_f1"] if early_stopping else val_f1


mlflow.set_tracking_uri(uri=args.mlflow_uri)
mlflow.set_experiment(args.mlflow_experiment)

pruner = optuna.pruners.MedianPruner(
    n_warmup_steps=20, n_startup_trials=10, n_min_trials=5, interval_steps=5
)
study = optuna.create_study(direction="maximize", pruner=pruner)
with mlflow.start_run(run_name=args.mlflow_run):
    study.optimize(objective, n_trials=args.trials)

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value:  {best_trial.value}")
    print(f"  Params: {best_trial.params}")

    mlflow.log_metric("best_val_loss", study.best_value)
    for key, value in best_trial.params.items():
        mlflow.log_param(f"best_{key}", value)

    print("Evaluating best model on test set...")
    model_path = best_trial.user_attrs["model_path"]
    model = model_factory(
        train_set,
        hidden_dim=best_trial.params["hidden_dim"],
        port_dim=2,
        dropout=best_trial.params["dropout"],
        nlayers=best_trial.params["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(model_path))

    mlflow.pytorch.log_model(model, "model")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_trial.params["lr"],
        weight_decay=best_trial.params["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )

    mlflow.log_metrics(
        {"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1}
    )
    utils.log_class_stats(all_preds, all_labels, suffix="test")

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}"
    )
