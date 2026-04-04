import yaml


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config