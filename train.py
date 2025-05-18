import argparse
import time

import mlflow
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.transforms import NormalizeFeatures

from train_common import train, evaluate, get_dataset_factory, get_model_factory
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single training and evaluation run")

    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("model", type=str, choices=["baseline", "repr1", "repr2"], help="Model to train")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    #parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "rmsprop"], help="Optimizer to use")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to run training on")

    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--tolerance", type=float, default=5e-3, help="Early stopping tolerance")

    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="xmatus36-gnns", help="MLflow experiment name")
    parser.add_argument("--mlflow-run", type=str, help="MLflow run name")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Model:", args.model)

    dset_factory = get_dataset_factory(args.model)
    transform = NormalizeFeatures()
    train_set = dset_factory(args.dataset_path, "train", transform=transform)
    val_set = dset_factory(args.dataset_path, "val", transform=transform)
    test_set = dset_factory(args.dataset_path, "test", transform=transform)

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

    print(f'Number of classes: {num_classes}')
    print("Class Weights:", class_weights)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = get_model_factory(args.model)(
        train_set,
        hidden_dim=args.hidden,
        port_dim=2,
        dropout=args.dropout,
        nlayers=args.layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = utils.EarlyStopping(patience=args.patience, tolerance=args.tolerance)

    all_preds = None
    all_labels = None
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run):
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_params(vars(args))

        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1, all_preds, all_labels = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }, step=epoch)

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
                print(f"Early stopping at epoch {epoch} with min val_loss: {early_stopping.best_loss:.4f}")
                break

        mlflow.log_metric('median_epoch_time', torch.tensor(times).median())
        if early_stopping:
            all_preds = early_stopping.data_at_best['all_preds']
            all_labels = early_stopping.data_at_best['all_labels']
            val_loss = early_stopping.data_at_best['val_loss']
            val_acc = early_stopping.data_at_best['val_acc']
            train_loss = early_stopping.data_at_best['val_loss']
            utils.log_class_stats(all_preds, all_labels, suffix="val")
        else:
            utils.log_class_stats(all_preds, all_labels, suffix="val")

        # Evaluate on the test set
        model.load_state_dict(early_stopping.best_model_state)
        test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(
            model, test_loader, criterion, device
        )

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        }, step=epoch)
        utils.log_class_stats(all_preds, all_labels, suffix="test")

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
