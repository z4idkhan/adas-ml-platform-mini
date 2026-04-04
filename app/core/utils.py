import os
import json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_json(file_path: str):
    with open(file_path, "r") as file:
        return json.load(file)