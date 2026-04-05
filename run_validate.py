from app.core.config import load_config
from app.data.validate import validate_dataset

def main():
    config = load_config()

    result = validate_dataset(
        config["paths"]["raw_data"],
        config["labels"]["classes"]
    )

    print("\nDataset Validation Result:\n")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()