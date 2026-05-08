import torch
import torch.nn as nn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from models.cnn import BasicCNN
from src.dataset import SettlementDataset, load_image_paths


def train_cnn(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths, labels = load_image_paths(
        config["data"]["formal_dir"],
        config["data"]["informal_dir"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths,
        labels,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=labels
    )

    train_dataset = SettlementDataset(
        X_train,
        y_train,
        image_size=config["training"]["image_size"],
        band_indices=config["model"]["band_indices"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    model = BasicCNN(
        input_channels=len(config["model"]["band_indices"])
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    model.train()

    for epoch in range(config["training"]["epochs"]):
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    model.eval()

    all_preds = []
    all_labels = []

    test_dataset = SettlementDataset(
        X_test,
        y_test,
        image_size=config["training"]["image_size"],
        band_indices=config["model"]["band_indices"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    print("Confusion Matrix:")
    print(cm)
    print(f"Test Accuracy: {accuracy:.4f}")

    Path(config["model"]["save_path"]).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), config["model"]["save_path"])

    print(f"Saved model to {config['model']['save_path']}")