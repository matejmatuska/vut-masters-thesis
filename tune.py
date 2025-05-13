import argparse
import os
import time

import mlflow
import optuna
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.transforms import NormalizeFeatures

from dataset import ChronoDataset, Repr1Dataset, SunDataset
from model.baseline import GraphClassifier
from model.chrono import ChronoClassifier
from model.repr1 import Repr1Classifier
import utils


def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(which, model, loader, criterion, device, log=False, f1_average='weighted'):
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(data.y.cpu())

            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if log:
        conf_matrix = confusion_matrix(all_labels, all_preds)
        utils.log_conf_matrix(conf_matrix, path=f"confusion_matrix_{which}.png")

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))

        report_dict = classification_report(all_labels, all_preds, output_dict=True)
        mlflow.log_dict(report_dict, artifact_file=f"classification_report_{which}.json")

    f1 = f1_score(all_labels, all_preds, average=f1_average)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, f1


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("model", type=str, choices=["baseline", "chrono", "repr1"], help="Model to train")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to run training on")

    parser.add_argument("--trials", type=int, default=30, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")

    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Early stopping tolerance")

    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="xmatus36-gnns", help="MLflow experiment name")
    parser.add_argument("--mlflow-run", type=str, help="MLflow main run name")
    return parser.parse_args()


def get_dataset_factory(which) -> (callable):
    if which == "baseline":
        def creator(root, split, **kwargs) -> SunDataset:
            return SunDataset(
                root,
                split,
                **kwargs,
            )
    elif which == "chrono":
        def creator(root, split, **kwargs) -> ChronoDataset:
            return ChronoDataset(
                root,
                split,
                **kwargs,
            )
    elif which == "repr1":
        def creator(root, split, **kwargs) -> Repr1Dataset:
            return Repr1Dataset(
                root,
                split,
                **kwargs,
            )
    else:
        raise ValueError("Invalid model type. Choose 'baseline', 'chrono' or 'repr1'.")
    return creator


def get_model_factory(which):
    if which == "baseline":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return GraphClassifier(
                edge_dim=dataset[0].edge_attr.size(1),
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif which == "chrono":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return ChronoClassifier(
                input_dim=dataset[0].num_node_features,
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    elif args.model == "repr1":
        def make_model(dataset, hidden_dim, dropout, nlayers=2):
            return Repr1Classifier(
                input_dim=97,
                num_hosts=0, # TODO unused
                hidden_dim=hidden_dim,
                num_classes=dataset.num_classes,
                layers=nlayers,
                dropout=dropout,
            )
    else:
        # should not reach here, handled by argparse
        raise ValueError("Invalid model type. Choose 'baseline', 'chrono' or 'repr1'.")
    return make_model


args = parse_args()
print("Model:", args.model)

dset_factory = get_dataset_factory(args.model)
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

num_classes = train_set.num_classes

device = torch.device(args.device)
class_weights = utils.compute_class_weights(train_set).to(device)

print(f'Number of classes: {num_classes}')
print("Class Weights:", class_weights)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model_factory = get_model_factory(args.model)

model_dir = os.path.join("models")
os.makedirs(model_dir, exist_ok=True)


def objective(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 96, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = model_factory(
        train_set,
        hidden_dim=hidden_dim,
        dropout=dropout,
        nlayers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = utils.EarlyStopping(patience=args.patience, tolerance=args.tolerance)

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            'model': args.model,
            'dataset': args.dataset_path,
            'device': args.device,
            'batch_size': args.batch_size,
        })
        mlflow.log_params({
            'num_classes': num_classes,
            'lr': learning_rate,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'epochs': args.epochs,
        })
        # mlflow.log_params(vars(args))

        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(
                "val", model, val_loader, criterion, device, epoch == args.epochs
            )

            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }, step=epoch)

            trial.report(val_f1, epoch)

            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch} with val_f1: {val_f1}")
                raise optuna.exceptions.TrialPruned()

            log(Epoch=epoch, Loss=train_loss, Val=val_loss, Acc=val_acc, F1=val_f1)
            times.append(time.time() - start)

            if early_stopping(val_loss, model, val_f1=val_f1, epoch=epoch):
                print(f"Early stopping at epoch {epoch} with min val_loss: {early_stopping.best_loss}")
                break

        mlflow.log_metric('median_epoch_time', torch.tensor(times).median())

        model_state = early_stopping.best_model_state if early_stopping else model.state_dict()
        model_path = os.path.join(model_dir, f"model_{trial.number}.pth")
        torch.save(model_state, model_path)
        trial.set_user_attr("model_path", model_path)

    return early_stopping.data_at_best['val_f1'] if early_stopping else val_f1


mlflow.set_tracking_uri(uri=args.mlflow_uri)
mlflow.set_experiment(args.mlflow_experiment)

study = optuna.create_study(direction="maximize")
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
        dropout=best_trial.params["dropout"],
        nlayers=best_trial.params["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_trial.params["lr"],
        weight_decay=best_trial.params["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    test_loss, test_acc, test_f1 = evaluate("test", model, test_loader, criterion, device, True)

    mlflow.log_metrics({
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1
    })
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
