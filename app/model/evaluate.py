import torch
from torch.utils.data import DataLoader

from app.core.config import load_config
from app.core.utils import save_json, ensure_dir
from app.data.ingest import load_samples
from app.data.split import split_samples
from app.data.transforms import get_eval_transforms
from app.model.dataset import RoadRiskDataset
from app.model.model import build_model
from app.analysis.metrics import compute_metrics


def evaluate_model():
    config = load_config()

    raw_data_path = config["paths"]["raw_data"]
    model_output = config["paths"]["model_output"]
    metrics_output = config["paths"]["metrics_output"]
    class_names = config["labels"]["classes"]

    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_classes = config["training"]["num_classes"]

    samples = load_samples(raw_data_path, class_names)
    _, _, test_samples = split_samples(samples)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    test_dataset = RoadRiskDataset(
        test_samples,
        class_to_idx,
        transform=get_eval_transforms(image_size)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_output, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_pred)

    ensure_dir("artifacts/metrics")
    save_json(metrics, metrics_output)

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    evaluate_model()