import torch
from PIL import Image

from app.core.config import load_config
from app.data.transforms import get_eval_transforms
from app.model.model import build_model


class Predictor:
    def __init__(self):
        self.config = load_config()
        self.class_names = self.config["labels"]["classes"]
        self.image_size = self.config["training"]["image_size"]
        self.num_classes = self.config["training"]["num_classes"]
        self.model_output = self.config["paths"]["model_output"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.num_classes)
        self.model.load_state_dict(torch.load(self.model_output, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_eval_transforms(self.image_size)

    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            prediction = torch.argmax(outputs, dim=1).item()

        return self.class_names[prediction]