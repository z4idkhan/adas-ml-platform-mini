import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from app.core.config import load_config
from app.core.utils import ensure_dir
from app.data.ingest import load_samples
from app.data.split import split_samples
from app.data.transforms import get_train_transforms, get_eval_transforms
from app.model.dataset import RoadRiskDataset
from app.model.model import build_model


def train_model():
    config = load_config()

    raw_data_path = config["paths"]["raw_data"]
    model_output = config["paths"]["model_output"]
    class_names = config["labels"]["classes"]

    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    num_classes = config["training"]["num_classes"]

    samples = load_samples(raw_data_path, class_names)
    train_samples, val_samples, _ = split_samples(samples)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    train_dataset = RoadRiskDataset(
        train_samples,
        class_to_idx,
        transform=get_train_transforms(image_size)
    )

    val_dataset = RoadRiskDataset(
        val_samples,
        class_to_idx,
        transform=get_eval_transforms(image_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    ensure_dir("artifacts/models")
    torch.save(model.state_dict(), model_output)
    print(f"Model saved to {model_output}")


if __name__ == "__main__":
    train_model()