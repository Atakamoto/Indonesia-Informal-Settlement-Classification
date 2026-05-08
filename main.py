import argparse
import yaml

from src.train import train_cnn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--model", default="cnn")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.model == "cnn":
        train_cnn(config)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()