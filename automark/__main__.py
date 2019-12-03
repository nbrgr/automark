import argparse
from automark import preprocess

def main():
    ap = argparse.ArgumentParser("AutoMark")

    ap.add_argument("mode", choices=["preprocess", "train", "generate"], help="preprocess data, train a model, or generate markings")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    args = ap.parse_args()

    if args.mode == 'preprocess':
        preprocess(config=args.config_path)